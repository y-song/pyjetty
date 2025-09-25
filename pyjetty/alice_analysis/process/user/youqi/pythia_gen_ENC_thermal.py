#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np
import math

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.mputils.csubtractor import CEventSubtractor

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
class PythiaGenENCThermal(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenENCThermal, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jetR_list = config["jetR"] 

        self.nev = args.nev

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

        if 'rm_trk_min_pt' in config:
            self.rm_trk_min_pt = config['rm_trk_min_pt']
        else:
            self.rm_trk_min_pt = False

        if 'jet_matching_distance' in config:
            self.jet_matching_distance = config['jet_matching_distance']
        else:
            self.jet_matching_distance = 0.6 # default to 0.6

        if 'mc_fraction_threshold' in config:
            self.mc_fraction_threshold = config['mc_fraction_threshold']
        else:
            self.mc_fraction_threshold = 0.5 # default to 0.5

        # perp cone settings
        if 'static_perpcone' in config:
            self.static_perpcone = config['static_perpcone']
        else:
            self.static_perpcone = True # NB: set default to rigid cone (less fluctuations)

        # perp and jet cone sizes
        self.coneR_list = config["coneR"] 

        # ENC settings
        if 'thrd' in config:
            self.thrd_list = config['thrd']
        else:
            self.thrd_list = [1.0]
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = 2
        self.npower = 1

        # thermal background
        if 'thermal_model' in config:
            self.thermal_model = True
            beta = config['thermal_model']['beta']
            N_avg = config['thermal_model']['N_avg']
            sigma_N = config['thermal_model']['sigma_N']
            self.thermal_generator = thermal_generator.ThermalGenerator(N_avg, sigma_N, beta)
        else:
            self.thermal_model = False

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
        
        self.scale_print_final_info(pythia)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        self.pair_type_labels = ['_bb','_sb','_ss']

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            R_label = str(jetR).replace('.', '') + 'Scaled'

            name = 'h_matched_area_JetPt_ch_R{}'.format(R_label)
            print('Initialize histogram',name)
            pt_bins = linbins(0,200,200)
            area_bins = linbins(0,1,100)
            h = ROOT.TH2D(name, name, 200, pt_bins, 100, area_bins)
            h.GetXaxis().SetTitle('p_{T, pp jet}')
            h.GetYaxis().SetTitle('Area')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            name = 'h_matched_JetPt_ch_combined_vs_pp_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH2D(name, name, 500, pt_bins, 500, pt_bins)
            h.GetXaxis().SetTitle('p_{T, comb jet}')
            h.GetYaxis().SetTitle('p_{T, pp jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            name = 'h_matched_JetPt_ch_JES_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            JES_bins = linbins(-1,1,200)
            h = ROOT.TH2D(name, name, 500, pt_bins, 200, JES_bins)     
            h.GetXaxis().SetTitle('p_{T, pp jet}')
            h.GetYaxis().SetTitle('(p_{T, comb jet}-p_{T, pp jet})/p_{T, pp jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            name = 'h_JetPt_ch_pp_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, pp jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            name = 'h_JetPt_ch_combined_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, comb jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            # histograms for local energy density related checks
            for observable in ['rho_local', 'mult']:
                for thrd in self.thrd_list:

                    thrd_label = 'trk{:.0f}'.format(thrd*10)
                    
                    if observable == 'rho_local':
                        obs_nbins = 100
                        obs_bins = linbins(0,500,obs_nbins)
                    else:
                        obs_nbins = 50
                        obs_bins = linbins(0,50,obs_nbins)
                    pt_bins = linbins(0,200,200)

                    name = 'h_matched_{}_JetPt_ch_R{}_{}'.format(observable, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                    h.GetXaxis().SetTitle('p_{T, pp jet}')
                    h.GetYaxis().SetTitle(observable)
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_matched_{}_JetPt_ch_combined_R{}_{}'.format(observable, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
                    h.GetYaxis().SetTitle(observable)
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    for coneR in self.coneR_list:
                        name = 'h_perpcone{}_matched_{}_JetPt_ch_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, pp jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_perpcone{}_matched_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_jetcone{}_matched_{}_JetPt_ch_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, pp jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_jetcone{}_matched_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

            # ENC histograms
            for ipoint in range(2, self.npoint+1):

                for thrd in self.thrd_list:

                    thrd_label = 'trk{:.0f}'.format(thrd*10)

                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)

                    name = 'h_ENC{}_JetPt_ch_R{}_{}'.format(str(ipoint), R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, pp jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint), R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    for pair_type_label in self.pair_type_labels:

                        name = 'h_matched_ENC{}_JetPt_ch_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, pp jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_matched_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_matched_ENC{}_JetPt_ch_mix_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, pp jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        for coneR in self.coneR_list:
                            name = 'h_perpcone{}_matched_ENC{}_JetPt_ch_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_perpcone{}_matched_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, comb jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_perpcone{}_matched_ENC{}_JetPt_ch_mix_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_jetcone{}_matched_ENC{}_JetPt_ch_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_jetcone{}_matched_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, comb jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_jetcone{}_matched_ENC{}_JetPt_ch_mix_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
            # NB: area calculation enabled
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, "jet_def_R%s" % jetR_str, jet_def)
            print(jet_def)

        # pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
        if self.rm_trk_min_pt:
            track_selector_ch = fj.SelectorPtMin(0)
        else:
            track_selector_ch = fj.SelectorPtMin(0.15)

        setattr(self, "track_selector_ch", track_selector_ch)

        pfc_selector1 = fj.SelectorPtMin(1.)
        setattr(self, "pfc_def_10", pfc_selector1)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

    #---------------------------------------------------------------
    # Analyze events and pass information on to jet finding
    #---------------------------------------------------------------
    def analyze_events(self, pythia):
        
        iev = 0  # Event loop count

        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event
            # print(self.event)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Add thermal particles to the list
            # NB: the thermal tracks are each stored with a unique user_index < 0
            self.fj_particles_combined_beforeCS = self.thermal_generator.load_event()
          
            # Add pythia particles to the list
            [self.fj_particles_combined_beforeCS.push_back(p) for p in self.parts_pythia_ch]

            if self.debug_level > 1:
                for p in self.fj_particles_combined_beforeCS:
                    print('particle info user_index',p.user_index(),'pt',p.perp(),'phi',p.phi(),'eta',p.eta(),)

            self.constituent_subtractor = CEventSubtractor(max_distance=self.max_distance, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR)
            self.constituent_subtractor.process_event(self.fj_particles_combined_beforeCS)

            self.rho = self.constituent_subtractor.bge_rho.rho() 

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)

            self.analyze_jets()

            iev += 1

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def analyze_jets(self):
        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            track_selector_ch = getattr(self, "track_selector_ch")

            cs_pp = fj.ClusterSequence(track_selector_ch(self.parts_pythia_ch), jet_def)
            jets_pp = fj.sorted_by_pt( jet_selector(cs_pp.inclusive_jets()) )

            cs_combined = fj.ClusterSequenceArea(track_selector_ch(self.fj_particles_combined_beforeCS), jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
            jets_combined = fj.sorted_by_pt( jet_selector(cs_combined.inclusive_jets()) )

            #-------------------------------------------------------------
            # match pp (pythia) jets to combined jets
            jets_combined_matched_to_pp = []
            for jet_pp in jets_pp:
                matched_jet_combined = []
                for index_jet_combined, jet_combined in enumerate(jets_combined):
                    mc_fraction = self.mc_fraction(jet_pp, jet_combined)
                    if (mc_fraction > self.mc_fraction_threshold) and (self.is_geo_matched(jet_combined, jet_pp, jetR)):
                        matched_jet_combined.append(index_jet_combined)
                    
                if len(matched_jet_combined)==1: # accept if there is one match only (NB: but may be used multiple times)
                    jets_combined_matched_to_pp.append(matched_jet_combined[0]) # save matched combined jet index
                    if self.debug_level > 0:
                        print('matched pp jet R',jetR,'pt',jet_pp.perp(),'phi',jet_pp.phi(),'eta',jet_pp.eta())
                        print('matched combined jet index',matched_jet_combined[0],'pt',jets_combined[matched_jet_combined[0]].perp(),'phi',jets_combined[matched_jet_combined[0]].phi(),'eta',jets_combined[matched_jet_combined[0]].eta())     
                else:
                    jets_combined_matched_to_pp.append(-1) 

            R_label = str(jetR).replace('.', '') + 'Scaled'

            #-------------------------------------------------------------
            # loop over jets and fill EEC histograms with jet constituents
            for jet_pp in jets_pp:
                hname = 'h_JetPt_ch_pp_R{}'.format(R_label)
                getattr(self, hname).Fill(jet_pp.perp())
                hname = 'h_ENC{{}}_JetPt_ch_R{}_{{}}'.format(R_label)
                self.fill_jet_histograms(hname, jet_pp)

            for jet_combined in jets_combined:
                hname = 'h_JetPt_ch_combined_R{}'.format(R_label)
                getattr(self, hname).Fill(jet_combined.perp()-self.rho*jet_combined.area())
                hname = 'h_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
                self.fill_jet_histograms(hname, jet_combined)

            #-------------------------------------------------------------
            # loop over matched jets and fill EEC histograms with jet constituents
            nmatched_pp = 0
            for index_jet_pp, jet_pp in enumerate(jets_pp):
                imatched_jet_combined = jets_combined_matched_to_pp[index_jet_pp]
                if imatched_jet_combined > -1:
                    nmatched_pp += 1
                    jet_combined = jets_combined[imatched_jet_combined]
                    self.fill_matched_jets(jet_combined, jet_pp, jetR)
                    for coneR in self.coneR_list:
                        self.fill_matched_jetcone(jet_combined, jet_pp, jetR, coneR)
                        if coneR == jetR:
                            use_constituents = True
                        else:
                            use_constituents = False
                        self.fill_matched_perpcone(jet_combined, jet_pp, jetR, coneR, use_constituents)
                        
                    hname = 'h_matched_JetPt_ch_combined_vs_pp_R{}'.format(R_label)
                    getattr(self, hname).Fill(jet_combined.perp()-self.rho*jet_combined.area(), jet_pp.perp())
                    hname = 'h_matched_JetPt_ch_JES_R{}'.format(R_label)
                    getattr(self, hname).Fill(jet_pp.perp(), (jet_combined.perp()-self.rho*jet_combined.area()-jet_pp.perp())/jet_pp.perp())

            if self.debug_level > 0:
                if len(jets_pp)>0:
                    print('matching efficiency:',nmatched_pp/len(jets_pp),'=',nmatched_pp,'/',len(jets_pp))
                else:
                    print('matching efficiency:',nmatched_pp,'/',len(jets_pp))

    #---------------------------------------------------------------
    # Fill jet constituents for unmatched jets
    #---------------------------------------------------------------
    def fill_jet_histograms(self, hname, jet):

        constituents = fj.sorted_by_pt(jet.constituents())

        for thrd in self.thrd_list:
            c_select = fj.vectorPJ()
            thrd_label = 'trk{:.0f}'.format(thrd*10)
            for c in constituents:
              if c.pt() < thrd:
                break
              c_select.append(c) # NB: use the break statement since constituents are already sorted

            dphi_cut = -9999
            deta_cut = -9999
            new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), self.npoint, self.npower, dphi_cut, deta_cut)

            for ipoint in range(2, self.npoint+1):
                for index in range(new_corr.correlator(ipoint).rs().size()):              
                    getattr(self,hname.format(ipoint, thrd_label)).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Fill perp cone for matched combined jets
    #---------------------------------------------------------------
    def fill_matched_jets(self, jet_combined, jet_pp, jetR):

        R_label = str(jetR).replace('.', '') + 'Scaled'

        # fill EEC for matched comb jet using pp jet for jet pT
        hname = 'h_matched_ENC{{}}_JetPt_ch_R{}_{{}}'.format(R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, None)

        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h_matched_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, None)

        # fill EEC for matched comb jet pp jet for jet pT selection and comb jet for energy weight
        hname = 'h_matched_ENC{{}}_JetPt_ch_mix_R{}_{{}}'.format(R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, None)

        hname = 'h_matched_area_JetPt_ch_R{}'.format(R_label)
        if self.debug_level > 0:
            print('area',jet_combined.area(),'rho',self.rho,'matche pp jet pt',jet_pp.perp())
        getattr(self, hname).Fill(jet_pp.perp(), jet_combined.area())

        # fill EEC for matched comb jet using pp jet for jet pT
        hname = 'h_matched_{{}}_JetPt_ch_R{}_{{}}'.format(R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, jetR, None)

        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h_matched_{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, jetR, None)
            
    #---------------------------------------------------------------
    # Fill perp cone for matched combined jets
    #---------------------------------------------------------------
    def fill_matched_perpcone(self, jet_combined, jet_pp, jetR, coneR, use_constituents):

        R_label = str(jetR).replace('.', '') + 'Scaled'

        perp_jet1 = fj.PseudoJet()
        perp_jet1.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi() + np.pi/2, jet_combined.m())
        perp_jet2 = fj.PseudoJet()
        perp_jet2.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi() - np.pi/2, jet_combined.m())

        #================================================================
        #   Bigger cones than AK jet R implemented
        # if use_constituents is true, use jet constituents for jet particles
        # else use particles in jet cone for jet particles, and use 
        # the same cone size for jet cone and perp cone 
        # NB: don't use constituents when coneR > jetR
        #================================================================
        perpcone_R = coneR
        # NB1: only enable dynamic option when coneR = jetR
        # NB2: similar result using dynamic and static cone
        if self.static_perpcone == False and coneR == jetR:
            perpcone_R = math.sqrt(jet_combined.area()/np.pi)
        
        if use_constituents:
            constituents = jet_combined.constituents()
            parts_in_jet = self.copy_parts(constituents) # NB: make a copy so that the original jet constituents will not be modifed
        else:
            parts_in_jet = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined, perpcone_R)

        # NB: a deep copy of the combined particle list are made before re-labeling the particle user_index (copy created in find_parts_around_jet) and assembling the perp cone parts
        parts_in_perpcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet1, perpcone_R)
        parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -np.pi/2)
          
        parts_in_perpcone2 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet2, perpcone_R)
        parts_in_perpcone2 = self.rotate_parts(parts_in_perpcone2, +np.pi/2)
        
        # use 999 and -999 to distinguish from prevous used labeling numbers
        parts_in_cone1 = fj.vectorPJ()
        # fill parts from jet
        for part in parts_in_jet:
          part.set_user_index(999)
          parts_in_cone1.append(part)
        # fill parts from perp cone 1
        for part in parts_in_perpcone1:
          part.set_user_index(-999)
          parts_in_cone1.append(part)
        
        parts_in_cone2 = fj.vectorPJ()
        # fill parts from jet
        for part in parts_in_jet:
          part.set_user_index(999)
          parts_in_cone2.append(part)
        # fill parts from perp cone 2
        for part in parts_in_perpcone2:
          part.set_user_index(-999)
          parts_in_cone2.append(part)
          
        # fill EEC for matched comb jet using pp jet for jet pT
        hname = 'h_perpcone{}_matched_ENC{{}}_JetPt_ch_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone1)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone2)

        # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
        hname = 'h_perpcone{}_matched_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone1)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone2)

        # fill EEC for matched comb jet pp jet for jet pT selection and comb jet for energy weight
        hname = 'h_perpcone{}_matched_ENC{{}}_JetPt_ch_mix_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone1)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_cone2)

        # fill rho local for matched comb jet using pp jet for jet pT
        hname = 'h_perpcone{}_matched_{{}}_JetPt_ch_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_cone1)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_cone2)

        # fill rho local for matched comb jet using comb jet (rho subtracted) for jet pT
        hname = 'h_perpcone{}_matched_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_cone1)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_cone2)
    
    #---------------------------------------------------------------
    # Fill jet cone for matched combined jets
    #---------------------------------------------------------------
    def fill_matched_jetcone(self, jet_combined, jet_pp, jetR, coneR):

        R_label = str(jetR).replace('.', '') + 'Scaled'

        jetcone_R = coneR

        parts_in_jetcone = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined, jetcone_R)
          
        # fill EEC for matched comb jet using pp jet for jet pT
        hname = 'h_jetcone{}_matched_ENC{{}}_JetPt_ch_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_jetcone)

        # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
        hname = 'h_jetcone{}_matched_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_ENC_histograms(hname, jet_pp, jet_combined, parts_in_jetcone)

        # fill rho local for matched comb jet using pp jet for jet pT
        hname = 'h_jetcone{}_matched_{{}}_JetPt_ch_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_jetcone)

        # fill rho local for matched comb jet using comb jet (rho subtracted) for jet pT
        hname = 'h_jetcone{}_matched_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_matched_rho_local_histograms(hname, jet_pp, jet_combined, coneR, parts_in_jetcone)

    #---------------------------------------------------------------
    # Fill matched ENC histograms
    #---------------------------------------------------------------
    def fill_matched_ENC_histograms(self, hname, jet_pp, jet_combined, cone_parts):
        
        if cone_parts == None:
            constituents = fj.sorted_by_pt(jet_combined.constituents())
        else:
            constituents = fj.sorted_by_pt(cone_parts)

        for thrd in self.thrd_list:
            c_select = fj.vectorPJ()
            thrd_label = 'trk{:.0f}'.format(thrd*10)
            for c in constituents:
              if c.pt() < thrd:
                break
              c_select.append(c) # NB: use the break statement since constituents are already sorted

            if 'combined' in hname:
                jet_pt_weight = jet_combined.perp()-self.rho*jet_combined.area()
                jet_pt_select = jet_combined.perp()-self.rho*jet_combined.area()
            elif 'mix' in hname:
                jet_pt_weight = jet_combined.perp()-self.rho*jet_combined.area()
                jet_pt_select = jet_pp.perp()
            else:
                jet_pt_weight = jet_pp.perp()
                jet_pt_select = jet_pp.perp()

            new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt_weight, self.npoint, self.npower, self.dphi_cut, self.deta_cut) # NB: using the pp jet as reference for energy weight

            for ipoint in range(2, self.npoint+1):
                for index in range(new_corr.correlator(ipoint).rs().size()):
                    pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
                    pair_type_label = self.pair_type_labels[pair_type]
                  
                    getattr(self, hname.format(str(ipoint) + pair_type_label,thrd_label)).Fill(jet_pt_select, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Fill matched rho local histograms
    #---------------------------------------------------------------
    def fill_matched_rho_local_histograms(self, hname, jet_pp, jet_combined, coneR, cone_parts):
        
        if cone_parts == None:
            constituents = fj.sorted_by_pt(jet_combined.constituents())
        else:
            constituents = fj.sorted_by_pt(cone_parts)

        if 'combined' in hname:
            jet_pt_select = jet_combined.perp()-self.rho*jet_combined.area()
        else:
            jet_pt_select = jet_pp.perp()

        for thrd in self.thrd_list:
            c_select = fj.vectorPJ()
            thrd_label = 'trk{:.0f}'.format(thrd*10)
            for c in constituents:
              if c.pt() < thrd:
                break
              c_select.append(c) # NB: use the break statement since constituents are already sorted
            
            pt_sum = 0
            N_sum = 0
            for c in c_select:
                if c.user_index() < 0:
                    pt_sum += c.perp()
                    N_sum += 1

            if 'jetcone' in hname or 'perpcone' in hname:
                jet_area = np.pi*coneR*coneR
            else:
                jet_area = jet_combined.area()
            getattr(self, hname.format('rho_local', thrd_label)).Fill(jet_pt_select, pt_sum/jet_area)
            getattr(self, hname.format('mult', thrd_label)).Fill(jet_pt_select, N_sum)
    
    #---------------------------------------------------------------
    # Compare two jets and store matching candidates in user_info
    #---------------------------------------------------------------
    def is_geo_matched(self, jet1, jet2, jetR):
        deltaR = jet1.delta_R(jet2)
      
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < self.jet_matching_distance * jetR:
            return True
        else:
            return False

    #---------------------------------------------------------------
    # Return pt-fraction of tracks in jet_pp that are contained in jet_combined
    #---------------------------------------------------------------
    def mc_fraction(self, jet_pp, jet_combined):

        pt_total = jet_pp.pt()
       
        pt_contained = 0.
        for track in jet_combined.constituents():
          if track.user_index() >= 0:
            pt_contained += track.pt()
               
        return pt_contained/pt_total

    #---------------------------------------------------------------
    # Select particles around jet axis
    #---------------------------------------------------------------
    def find_parts_around_jet(self, parts, jet, coneR):

        cone_parts = fj.vectorPJ()
        for part in parts:
          if jet.delta_R(part) <= coneR:
            cone_parts.push_back(part)
        
        return cone_parts

    #---------------------------------------------------------------
    # Rotate parts in azimuthal direction 
    #---------------------------------------------------------------
    def rotate_parts(self, parts, rotate_phi):

        parts_rotated = fj.vectorPJ()
        for part in parts:
          pt_new = part.pt()
          y_new = part.rapidity()
          phi_new = part.phi() + rotate_phi
          m_new = part.m()
          user_index_new = part.user_index() # NB: manually update the user index
          # print('before',part.phi())
          part.reset_PtYPhiM(pt_new, y_new, phi_new, m_new)
          part.set_user_index(user_index_new)
          # print('after',part.phi())
          parts_rotated.push_back(part)

        return parts_rotated

    #---------------------------------------------------------------
    # Create a copy of list of particles
    #---------------------------------------------------------------
    def copy_parts(self, parts, remove_ghosts = True):
    # don't need to re-init every part for a deep copy
    # the last arguement enable/disable the removal of ghost particles from jet area calculation (default set to true)
        parts_copied = fj.vectorPJ()
        for part in parts:
          if remove_ghosts:
            if part.pt() > 0.01:
              parts_copied.push_back(part)
          else:
            parts_copied.push_back(part)

        return parts_copied

    #---------------------------------------------------------------
    # Detemine pair type (ss, sb, bb)
    #---------------------------------------------------------------
    def check_pair_type(self, corr_builder, ipoint, constituents, index):

        part1 = int(corr_builder.correlator(ipoint).indices1()[index])
        part2 = int(corr_builder.correlator(ipoint).indices2()[index])
        type1 = constituents[part1].user_index()
        type2 = constituents[part2].user_index()

        # NB: match the strings in self.pair_type_label = ['bb','sb','ss']
        if type1 < 0 and type2 < 0:
          # print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp()
          return 0 # means bkg-bkg
        if type1 < 0 and type2 >= 0:
          # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
          return 1 # means sig-bkg
        if type1 >= 0 and type2 < 0:
          # print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
          return 1 # means sig-bkg
        if type1 >= 0 and type2 >= 0:
          # print('sig-sig (',type1,type2,') pt1',constituents[part1].perp()
          return 2 # means sig-sig

    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print("scaling factor is",scale_f)

        for jetR in self.jetR_list:
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '') 
            for h in getattr(self, hist_list_name):
                h.Scale(scale_f)

        print("N total final events:", int(self.hNevents.GetBinContent(1)), "with",
              int(pythia.info.nAccepted() - self.hNevents.GetBinContent(1)),
              "events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)

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

    process = PythiaGenENCThermal(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)