#!/usr/bin/env python
# Example usage:
# python /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/process/user/youqi/pythia_gen_tree.py -c /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/pyjetty/alice_analysis/config/ENC/pp/pythia_gen.yaml --nev 20 --py-pthatmin 12 --py-ecm 5360 --py-seed 1

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
import ecorrel # /global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/cpptools/src/ecorrel
from heppy.pythiautils import configuration as pyconf # /global/cfs/cdirs/alice/youqi/mypyjetty/heppy/heppy/pythiautils/configuration.py
from pyjetty.mputils import *
from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class PythiaGenTree(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenTree, self).__init__(
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

        # Create ROOT TFile for storing TTree
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        # Create TTrees
        self.eec_tree = ROOT.TTree("EEC_Tree", "EEC Tree")
        self.shape_tree = ROOT.TTree("Shape_Tree", "Shape Tree")

        # Create variables for branches as instance attributes
        self.jet_pt = array.array('d', [0.0])
        self.pair_rl = array.array('d', [0.0])
        self.pair_weight = array.array('d', [0.0])
        self.part_r = array.array('d', [0.0])
        self.part_weight = array.array('d', [0.0])

        self.eec_tree.Branch("jet_pt", self.jet_pt, "jet_pt/D")
        self.eec_tree.Branch("pair_rl", self.pair_rl, "pair_rl/D")
        self.eec_tree.Branch("pair_weight", self.pair_weight, "pair_weight/D")
        self.shape_tree.Branch("jet_pt", self.jet_pt, "jet_pt/D")
        self.shape_tree.Branch("part_r", self.part_r, "part_r/D")
        self.shape_tree.Branch("part_weight", self.part_weight, "part_weight/D")

        mycfg = []
        self.pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.analyze_events()
        self.pythia.stat()
        print()

        outf.Write()
        outf.Close()

    #---------------------------------------------------------------
    # Initiate jet defs and selectors
    #---------------------------------------------------------------
    def init_jet_tools(self):

        if self.rm_trk_min_pt:
            self.track_selector_ch = fj.SelectorPtMin(0)
        else:
            self.track_selector_ch = fj.SelectorPtMin(0.15)

        for jetR in self.jetR_list:
            self.jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            print(self.jet_def)

            self.jet_selector = fj.SelectorPtMin(self.jet_min_pt) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)

    #---------------------------------------------------------------
    # Analyze events and jets
    #---------------------------------------------------------------
    def analyze_events(self):

        iev = 0  # Event loop count

        while iev < self.nev:

            if self.debug_level > 0:
                print('ievt', iev)
            if iev % 10000 == 0:
                print('ievt', iev)

            if not self.pythia.next():
                continue

            self.event = self.pythia.event

            iev += 1

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(self.pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            cs = fj.ClusterSequence( self.track_selector_ch(self.parts_pythia_ch), self.jet_def )
            jets = fj.sorted_by_pt( self.jet_selector(cs.inclusive_jets()) )

            for jet in jets:
                self.fill_trees(jet)

    #---------------------------------------------------------------
    # Fill EEC TTree
    #---------------------------------------------------------------
    def fill_trees(self, jet):

        constituents = fj.sorted_by_pt(jet.constituents())

        for thrd in self.thrd_list:

            c_select = fj.vectorPJ()
            for c in constituents:
                if c.pt() < thrd:
                    break
                c_select.append(c) # NB: use the break statement since constituents are already sorted

            new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), 2, 1, -9999, -9999)

            ipoint = 2
            correlator_rs = new_corr.correlator(ipoint).rs()
            correlator_weights = new_corr.correlator(ipoint).weights()

            for index in range(correlator_rs.size()):
                self.jet_pt[0] = jet.perp()
                self.pair_rl[0] = correlator_rs[index]
                self.pair_weight[0] = correlator_weights[index]
                self.eec_tree.Fill()

            for part in c_select:
                self.jet_pt[0] = jet.perp()
                self.part_r[0] = jet.delta_R(part)
                self.part_weight[0] = part.perp()/jet.perp()
                self.shape_tree.Fill()

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly - TTree version',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./',
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults_tree.root", type=str,
                        help="Filename for the generated particle ROOT TTree")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.config_file))
        sys.exit(0)

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    process = PythiaGenTree(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)