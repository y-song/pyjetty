#!/usr/bin/env python
# Example usage:
# python /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/youqi/pythia_gen.py -c /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/config/ENC/pp/pythia_gen.yaml --nev 20 --py-pthatmin 28 --py-seed 1

from __future__ import print_function

import yaml
import argparse
import os
import array
import numpy as np
import sys

import fastjet as fj
import ROOT

import pythiafjext # /global/cfs/cdirs/alice/youqi/mypyjetty/heppy/cpptools/src/pythiafjext/
import ecorrel
from heppy.pythiautils import configuration as pyconf
from pyjetty.mputils import *
from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class PythiaGen(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGen, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Set parameters
        self.jetR_list = config["jetR"] 
        self.jet_min_pt = config["jet_min_pt"]
        self.nev = args.nev
        self.max_eta_hadron = 0.9

        if 'rm_trk_min_pt' in config:
            self.rm_trk_min_pt = config['rm_trk_min_pt']
        else:
            self.rm_trk_min_pt = False

        # ENC settings
        if 'thrd' in config:
            self.thrd_list = config['thrd']
        else:
            self.thrd_list = [1.0]

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = []
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.analyze_events(pythia)
        pythia.stat()
        print()
        
        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        pt_bins = linbins(0,200,200)
        RL_bins = logbins(1E-4,1,50)
    
        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            jetR_str = str(jetR).replace('.', '')

            name = 'h_JetPt_R{}'.format(jetR_str)
            h = ROOT.TH1D(name, name, 200, pt_bins)
            h.GetXaxis().SetTitle('p_{T, jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)
            
            name = 'h_EEC_JetPt_R{}'.format(jetR_str)
            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
            h.GetXaxis().SetTitle('p_{T, jet}')
            h.GetYaxis().SetTitle('R_{L}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs and selectors
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        if self.rm_trk_min_pt:
            self.track_selector_ch = fj.SelectorPtMin(0)
        else:
            self.track_selector_ch = fj.SelectorPtMin(0.15)

        for jetR in self.jetR_list:
            
            jetR_str = str(jetR).replace('.', '')
            
            self.jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            print(self.jet_def)
            
            self.jet_selector = fj.SelectorPtMin(self.jet_min_pt) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)

    #---------------------------------------------------------------
    # Analyze events and jets
    #---------------------------------------------------------------
    def analyze_events(self, pythia):
        
        jetR_str = str(self.jetR_list[0]).replace('.', '')

        iev = 0  # Event loop count

        while iev < self.nev:

            if self.debug_level > 0:
                print('ievt', iev)
            if iev % 100 == 0:
                print('ievt', iev)

            if not pythia.next():
                continue

            self.event = pythia.event

            iev += 1

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
            
            cs = fj.ClusterSequence( self.track_selector_ch(self.parts_pythia_ch), self.jet_def )
            jets = fj.sorted_by_pt( self.jet_selector(cs.inclusive_jets()) )
            
            for jet in jets:
        
                hname = 'h_JetPt_R{}'.format(jetR_str)
                getattr(self, hname).Fill(jet.perp())

                hname = 'h_EEC_JetPt_R{}'.format(jetR_str)
                self.fill_EEC_histograms(hname, jet)
            
    #---------------------------------------------------------------
    # Fill EEC histograms
    #---------------------------------------------------------------
    def fill_EEC_histograms(self, hname, jet):
        
        constituents = fj.sorted_by_pt(jet.constituents())
        
        for thrd in self.thrd_list:
            
            c_select = fj.vectorPJ()
            for c in constituents:
                if c.pt() < thrd:
                    break
                c_select.append(c) # NB: use the break statement since constituents are already sorted

            new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), 2, 1, -9999, -9999) 

            ipoint = 2
            for index in range(new_corr.correlator(ipoint).rs().size()):
                getattr(self, hname).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    process = PythiaGen(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)