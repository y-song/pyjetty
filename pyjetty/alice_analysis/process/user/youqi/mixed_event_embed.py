#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT
import uproot

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np
import math
import time
import sys
import pandas
import random
import gc

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext # /global/cfs/cdirs/alice/youqi/mypyjetty/heppy/cpptools/src/pythiafjext/
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.mputils.csubtractor import CEventSubtractor
from pyjetty.mputils.icsubtractor import ICEventSubtractor
from pyjetty.alice_analysis.process.base import process_io

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
class ProcessEmbedENC(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(ProcessEmbedENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if (args.inputFileMc):
            self.input_file_mc = args.inputFileMc

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
            self.jet_matching_distance = 0.6
        
        if 'mc_fraction_threshold' in config:
            self.mc_fraction_threshold = config['mc_fraction_threshold']
        else:
            self.mc_fraction_threshold = 0.5
        
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

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def embed(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.process_data()
        self.process_mc()
        self.analyze_events()
        self.scale_hist()
        print()
        
        outf.Write()
        outf.Close()

        self.save_output_objects()

        del self.df_fjparticles
        del self.df_fjparticles_mc
        gc.collect()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):
        
        fmult = ROOT.TFile("/global/cfs/cdirs/alice/vdoomra/multiplicity_dist_centralPbPb.root","READ")
        hmult_temp = fmult.Get("hmult")
        self.hmult = hmult_temp.Clone("hmult_clone")
        self.hmult.SetDirectory(0)  # Detach from file directory
        fmult.Close()
        
        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        self.pair_type_labels = ['_bb','_sb','_ss']

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            R_label = str(jetR).replace('.', '')

            name = 'h_JetPt_ch_combined_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, comb jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

            name = 'h_matched_JetPt_ch_combined_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, comb jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)
            
            # rho_local and mult histograms
            for observable in ['rho_local', 'mult']:
                for thrd in self.thrd_list:

                    thrd_label = 'trk{:.0f}'.format(thrd*10)
                    
                    if observable == 'rho_local':
                        obs_nbins = 120
                        obs_bins = linbins(0,600,obs_nbins)
                    else:
                        obs_nbins = 50
                        obs_bins = linbins(0,50,obs_nbins)
                    pt_bins = linbins(0,200,200)

                    name = 'h_{}_JetPt_ch_combined_R{}_{}'.format(observable, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
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
                        
                        # jet mbcone combined
                        name = 'h_mbcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)
                        
                        # matched jet mbcone combined
                        name = 'h_matched_mbcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

            # ENC histograms
            ipoint = 2
            for thrd in self.thrd_list:

                thrd_label = 'trk{:.0f}'.format(thrd*10)

                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)

                name = 'h_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint), R_label, thrd_label)
                print('Initialize histogram',name)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T, comb jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_matched_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint), R_label, thrd_label)
                print('Initialize histogram',name)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T, comb jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                for pair_type_label in self.pair_type_labels:

                    # truth information
                    
                    name = 'h_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
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

                    for coneR in self.coneR_list:

                        # jet mbcone combined
                        name = 'h_mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # matched jet mbcone combined
                        name = 'h_matched_mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        if (pair_type_label == '_sb'):

                            name = 'h_2mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)
                            
                            name = 'h_matched_2mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
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

            jet_def_wta = fj.JetDefinition(fj.cambridge_algorithm, 2*jetR)
            jet_def_wta.set_recombination_scheme(fj.WTA_pt_scheme)
            setattr(self, "jet_def_wta_R%s" % jetR_str, jet_def_wta)
            print(jet_def_wta)

        if self.rm_trk_min_pt:
            track_selector_ch = fj.SelectorPtMin(0)
        else:
            track_selector_ch = fj.SelectorPtMin(0.15)
        setattr(self, "track_selector_ch", track_selector_ch)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

            jet_selector_40 = fj.SelectorPtMin(40) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_40_R%s" % jetR_str, jet_selector_40)

    #---------------------------------------------------------------
    # Analyze events and pass information on to jet finding
    #---------------------------------------------------------------
    def analyze_events(self):
        
        jetR_str = str(self.jetR_list[0]).replace('.', '')
        jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
        jet_def = getattr(self, "jet_def_R%s" % jetR_str)
        track_selector_ch = getattr(self, "track_selector_ch")
        jet_pt_thrd = self.check_jet_pt_thrd()

        print("Nevt(PbPb):", self.nEvents)
        print("Nevt(MC): ", self.nEvents_mc)
        
        iev_mc = 0 # Event loop count
        self.used_ev_mask = np.zeros(self.nEvents, dtype=bool)

        while (iev_mc < self.nEvents_mc):
            
            # assuming they are charged final states...
            self.parts_pythia_ch = fj.vectorPJ(self.df_fjparticles_mc.iloc[iev_mc])

            self.cs_pp = fj.ClusterSequence(track_selector_ch(self.parts_pythia_ch), jet_def)
            self.jets_pp = fj.sorted_by_pt( jet_selector(self.cs_pp.inclusive_jets()) )
            
            # if leading jet pT is over the thrd for the given pTHat bin, go to next MC
            if (len(self.jets_pp) > 0 and self.jets_pp[0].perp() > jet_pt_thrd):
                print("skip due to high weight")
                iev_mc += 1
                continue

            # check MC event info and get iev for SE and ME candidates
            mc_vtx = self.df_evts_mc.iloc[iev_mc]["z_vtx_reco"]
            se_iev, me_iev, me2_iev = self.get_se_and_me(mc_vtx)

            if (se_iev == None):
                iev_mc += 1
                continue
            
            # read in a SE
            self.fj_particles_combined_beforeCS_temp = self.get_mixed_event(se_iev)
            bkg_counter = -1
            self.fj_particles_combined_beforeCS = fj.vectorPJ()
            for p in self.fj_particles_combined_beforeCS_temp:
                p.set_user_index(bkg_counter)
                bkg_counter -= 1
                self.fj_particles_combined_beforeCS.push_back(p)
            # read in a ME
            self.fj_particles_combined_beforeCS_mb1_temp = fj.vectorPJ(self.df_fjparticles.iloc[me_iev])
            bkg_counter = -1
            self.fj_particles_combined_beforeCS_mb1 = fj.vectorPJ()
            for p in self.fj_particles_combined_beforeCS_mb1_temp:
                p.set_user_index(bkg_counter)
                bkg_counter -= 1
                self.fj_particles_combined_beforeCS_mb1.push_back(p)            
            self.used_ev_mask[me_iev] = True
            # read in another ME
            self.fj_particles_combined_beforeCS_mb2_temp = fj.vectorPJ(self.df_fjparticles.iloc[me2_iev])
            bkg_counter = -1
            self.fj_particles_combined_beforeCS_mb2 = fj.vectorPJ()
            for p in self.fj_particles_combined_beforeCS_mb2_temp:
                p.set_user_index(bkg_counter)
                bkg_counter -= 1
                self.fj_particles_combined_beforeCS_mb2.push_back(p)
            self.used_ev_mask[me2_iev] = True

            # Add particles from all pythia jets to the list
            self.parts_pythia_ch_jet = fj.vectorPJ()
            sig_counter = 1
            for ijet in range(0, len(self.jets_pp)):
                for p in self.jets_pp[ijet].constituents():
                    p.set_user_index(sig_counter)
                    sig_counter += 1
                    self.parts_pythia_ch_jet.push_back(p)
            [self.fj_particles_combined_beforeCS.push_back(p) for p in self.parts_pythia_ch_jet]

            if (self.do_constituent_subtraction):
                self.constituent_subtractor = CEventSubtractor(max_distance=self.max_distance, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR)
            elif (self.do_ics):
                self.constituent_subtractor = ICEventSubtractor(max_distances=fjcontrib.vectorDouble(self.max_distances), alphas=fjcontrib.vectorDouble(self.alphas), max_eta=self.max_eta, jet_median_selector_R=self.jet_median_selector_R, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area)
            self.fj_particles_combined_afterCS = self.constituent_subtractor.process_event(self.fj_particles_combined_beforeCS)
            self.rho = self.constituent_subtractor.bge_rho.rho()

            self.hNevents.Fill(0)
            self.analyze_jets()

            iev_mc += 1

    #---------------------------------------------------------------
    # Take pp jets, and embed them into PbPb
    #---------------------------------------------------------------
    def analyze_jets(self):
            
        jetR_str = str(self.jetR_list[0]).replace('.', '')
        jet_def = getattr(self, "jet_def_R%s" % jetR_str)
        jet_def_wta = getattr(self, "jet_def_wta_R%s" % jetR_str)
        reclusterer_wta = fjcontrib.Recluster(jet_def_wta)
        track_selector_ch = getattr(self, "track_selector_ch")
        jet_selector_40 = getattr(self, "jet_selector_40_R%s" % jetR_str)
        area_cut = 0.6*np.pi*self.jetR_list[0]*self.jetR_list[0]
        
        cs_combined = fj.ClusterSequenceArea(track_selector_ch(self.fj_particles_combined_beforeCS), jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
        jets_combined_preselect = fj.sorted_by_pt( jet_selector_40(cs_combined.inclusive_jets()) )        
        
        jets_combined = []
        jets_combined_wta = []
        for jet_combined in jets_combined_preselect:
            if (jet_combined.area() < area_cut):
                continue
            jets_combined.append(jet_combined)
            jet_combined_wta = reclusterer_wta.result(jet_combined)
            jets_combined_wta.append(jet_combined_wta)

        if (len(jets_combined) == 0):
            return

        # match pp jets to combined jets
        jets_combined_matched_index = [-1 for x in range(0, len(jets_combined))]
        for ijet_pp in range(0, len(self.jets_pp)):
            jet_pp = self.jets_pp[ijet_pp]
            jet_combined_matched = []
            ijet_combined_matched = -1
            for ijet_combined in range(0, len(jets_combined)):
                jet_combined = jets_combined[ijet_combined]
                if (jets_combined_matched_index[ijet_combined] != -1): # combined jet already has a match
                    continue
                if (self.mc_fraction(jet_pp, jet_combined) > self.mc_fraction_threshold) and (self.is_geo_matched(jet_combined, jet_pp, self.jetR_list[0])):
                    jet_combined_matched.append(jet_combined)
                    ijet_combined_matched = ijet_combined
            if (len(jet_combined_matched) == 1): # pp jet has a unique combined jet match
                jets_combined_matched_index[ijet_combined_matched] = ijet_pp
                
        # Main jet loop
        for i in range(0, len(jets_combined)):
            
            jet_combined_wta = jets_combined_wta[i]
            jet_constituents_afterCS = self.find_parts_around_jet(self.fj_particles_combined_afterCS, jet_combined_wta, self.jetR_list[0])
            jet_constituents_beforeCS = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined_wta, self.jetR_list[0])
            if (len(jet_constituents_afterCS) == 0):
                continue

            jet_sub_pt = 0
            for p in jet_constituents_afterCS:
                jet_sub_pt += p.perp()

            R_label = str(self.jetR_list[0]).replace('.', '')
            hname = 'h_JetPt_ch_combined_R{}'.format(R_label)
            getattr(self, hname).Fill(jet_sub_pt)
            
            self.fill_jets(jet_constituents_beforeCS, jet_sub_pt, self.jetR_list[0])
            self.fill_mbcone(jet_constituents_beforeCS, jet_sub_pt, jet_combined_wta, self.jetR_list[0])
            self.fill_2mbcone(jet_constituents_beforeCS, jet_sub_pt, jet_combined_wta, self.jetR_list[0])

            if (jets_combined_matched_index[i] != -1):
                hname = 'h_matched_JetPt_ch_combined_R{}'.format(R_label)
                getattr(self, hname).Fill(jet_sub_pt)

                self.fill_jets(jet_constituents_beforeCS, jet_sub_pt, self.jetR_list[0], "_matched")
                self.fill_mbcone(jet_constituents_beforeCS, jet_sub_pt, jet_combined_wta, self.jetR_list[0], "_matched")
                self.fill_2mbcone(jet_constituents_beforeCS, jet_sub_pt, jet_combined_wta, self.jetR_list[0], "_matched")

    #---------------------------------------------------------------
    # Fill perp cone for matched combined jets
    #---------------------------------------------------------------
    def fill_jets(self, jet_constituents, jet_pt, jetR, hist_prefix=""):

        R_label = str(jetR).replace('.', '')
        
        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(hist_prefix, R_label)
        self.fill_ENC_histograms(hname, jet_pt, jet_constituents)

        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(hist_prefix, R_label)
        self.fill_rho_local_histograms(hname, jet_pt, jetR, jet_constituents)

    #---------------------------------------------------------------
    # Fill mb cone for matched combined jets
    #---------------------------------------------------------------
    def fill_mbcone(self, jet_constituents, jet_pt, jet_axis, jetR, hist_prefix=""):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        mbcone_R = jetR
        
        # Do mb cones for the E-scheme jet and E-scheme jet cone
        mb_jet1 = fj.PseudoJet()
        mb_jet1.reset_PtYPhiM(jet_axis.pt(), jet_axis.rapidity(), jet_axis.phi(), jet_axis.m())
        parts_in_mbcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb1, mb_jet1, mbcone_R)
        
        # use 999 and -999 to distinguish from previous used labeling numbers
        parts_in_cone1 = fj.vectorPJ()
        # fill parts from jet
        for part in jet_constituents: # everything in the jet cone is "signal"
            part.set_user_index(999)
            parts_in_cone1.append(part)
        # fill parts from mb cone 1
        for part in parts_in_mbcone1:
            part.set_user_index(-999)
            parts_in_cone1.append(part)

        # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
        hname = 'h{}_mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(hist_prefix, jetR, R_label)
        self.fill_ENC_histograms(hname, jet_pt, parts_in_cone1)

        hname = 'h{}_mbcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(hist_prefix, jetR, R_label)
        self.fill_rho_local_histograms(hname, jet_pt, jetR, parts_in_cone1)

    def fill_2mbcone(self, jet_constituents, jet_pt, jet_axis, jetR, hist_prefix=""):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        mbcone_R = jetR
        
        # Do MB cone for the E-scheme jet and E-scheme jet cone
        mb_jet1 = fj.PseudoJet()
        mb_jet1.reset_PtYPhiM(jet_axis.pt(), jet_axis.rapidity(), jet_axis.phi(), jet_axis.m())
        mb_jet2 = fj.PseudoJet()
        mb_jet2.reset_PtYPhiM(jet_axis.pt(), jet_axis.rapidity(), jet_axis.phi(), jet_axis.m())
        
        parts_in_mbcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb1, mb_jet1, mbcone_R)     
        parts_in_mbcone2 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb2, mb_jet2, mbcone_R)

        # label one perpcone as "sig" and the other as "bkg" so the perp1-perp2 and perp1(2)-perp1(2) correlations can be saved separately
        parts_in_cone = fj.vectorPJ()
        for part in parts_in_mbcone1:
            part.set_user_index(999)
            parts_in_cone.append(part)
        for part in parts_in_mbcone2:
            part.set_user_index(-999)
            parts_in_cone.append(part)
            
        hname = 'h{}_2mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(hist_prefix, jetR, R_label)
        self.fill_ENC_histograms(hname, jet_pt, parts_in_cone)

    #---------------------------------------------------------------
    # Fill matched ENC histograms
    #---------------------------------------------------------------
    def fill_ENC_histograms(self, hname, jet_pt, cone_parts):
        
        constituents = fj.sorted_by_pt(cone_parts)

        for thrd in self.thrd_list:
            c_select = fj.vectorPJ()
            thrd_label = 'trk{:.0f}'.format(thrd*10)
            for c in constituents:
                if c.pt() < thrd:
                    break
                c_select.append(c) # NB: use the break statement since constituents are already sorted
            # print("N(constituents) w/ thrd:", len(c_select))

            if 'combined' in hname:
                jet_pt_weight = jet_pt
                jet_pt_select = jet_pt

            new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt_weight, self.npoint, self.npower, self.dphi_cut, self.deta_cut) 

            ipoint = 2
            for index in range(new_corr.correlator(ipoint).rs().size()):
                pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
                pair_type_label = self.pair_type_labels[pair_type]
                  
                if (('2perpcone' in hname or '2mbcone' in hname) and pair_type_label != '_sb'):
                    continue
                getattr(self, hname.format(str(ipoint) + pair_type_label,thrd_label)).Fill(jet_pt_select, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Fill matched rho local histograms
    #---------------------------------------------------------------
    def fill_rho_local_histograms(self, hname, jet_pt, coneR, cone_parts):
        
        constituents = fj.sorted_by_pt(cone_parts)

        if 'combined' in hname:
            jet_pt_select = jet_pt

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

            jet_area = np.pi*coneR*coneR
            
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
        
        # print("len(jet.constituents()):", len(jet.constituents()))
        cone_parts = fj.vectorPJ()
        for part in parts:
          if jet.delta_R(part) <= coneR:
            cone_parts.push_back(part)
        # print("len(cone_parts):", len(cone_parts))

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
    def scale_hist(self):
        
        pt_hat_yaml_file = "/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC20g4_mcid/scaleFactors.yaml"
        pt_hat_bin = int(self.input_file_mc.split('/')[len(self.input_file_mc.split('/'))-4]) # depends on exact format of input_file name
        with open(pt_hat_yaml_file, 'r') as stream:
            pt_hat_yaml = yaml.safe_load(stream)
            pt_hat = pt_hat_yaml[pt_hat_bin]
            # print("pt hat bin : " + str(pt_hat_bin))
            # print("pt hat weight : " + str(pt_hat))

        for jetR in self.jetR_list:
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '') 
            for h in getattr(self, hist_list_name):
                h.Scale(pt_hat)
        
    #---------------------------------------------------------------
    # Get the maximum jet pT allowed given the event pt_hat
    #---------------------------------------------------------------
    def check_jet_pt_thrd(self):
        jet_pt_thrd_yaml_file = "/global/cfs/cdirs/alice/youqi/jet_pt_thrd.yaml"
        pt_hat_bin = int(self.input_file_mc.split('/')[len(self.input_file_mc.split('/'))-4]) # depends on exact format of input_file name
        jet_pt_thrd = 1000.0
        if (pt_hat_bin <= 10):
            with open(jet_pt_thrd_yaml_file, 'r') as stream:
                jet_pt_thrd_yaml = yaml.safe_load(stream)
                jet_pt_thrd = jet_pt_thrd_yaml[pt_hat_bin]
                print("pt hat bin : " + str(pt_hat_bin))
                print("jet pt thrd: " + str(jet_pt_thrd))
        return jet_pt_thrd

    #---------------------------------------------------------------
    # Construct a mixed event using tracks from various real events
    #---------------------------------------------------------------
    def get_mixed_event(self, se_iev):

        ntrks_target = (int)(self.hmult.GetRandom())
        ntrks_added = 0
        # iev_unique = []
        event_candidate_iev = 0
        
        mixed_event = fj.vectorPJ()
        mixed_event_vz = self.vz_array[se_iev]
        mixed_event_centrality = self.centrality_array[se_iev]

        while (ntrks_added < ntrks_target):

            event_candidate_iev = event_candidate_iev % (self.nEvents)
            event_candidate_vz = self.vz_array[event_candidate_iev]
            event_candidate_centrality = self.centrality_array[event_candidate_iev]

            if ( abs(event_candidate_vz - mixed_event_vz) > 1.0 or abs(event_candidate_centrality - mixed_event_centrality) > 2.0 ):
                event_candidate_iev = event_candidate_iev + 1
                continue
            
            event_select_particles = self.df_fjparticles.iloc[event_candidate_iev]
            event_select_ntrks = len(event_select_particles) # total number of tracks in the selected event
            event_select_itrk = random.randint(0, event_select_ntrks - 1)
            # print("event_select_itrk (out of event_select_ntrks):", event_select_ntrks, event_select_itrk)

            mixed_event.push_back(event_select_particles[event_select_itrk])
            ntrks_added += 1
            event_candidate_iev += 1
            # if (event_candidate_iev not in iev_unique):
            #     iev_unique.append(event_candidate_iev)
            # print("itrk, event_candidate_iev:", ntrks_added, event_candidate_iev)

        # if (ntrks_target % 100 == 0): # randomly check this
        #     print("ntrks, nevts_unique:", ntrks_target, len(iev_unique))

        return mixed_event

    #---------------------------------------------------------------
    # Get a triplet of event numbers with appropriate event topologies, one for SE and two for MEs
    #---------------------------------------------------------------
    def get_se_and_me(self, mc_vtx):
        
        se_mask = (np.abs(self.vz_array - mc_vtx) < 1.0)
        se_candidates = np.where(se_mask)[0]
        if (len(se_candidates) == 0):
            return None, None, None
        
        se_iev = se_candidates[0]
        se_vtx = self.vz_array[se_iev]
        se_cent = self.centrality_array[se_iev]

        me_mask = ( (np.abs(self.centrality_array - se_cent) < 2.0) & (np.abs(self.vz_array - se_vtx) < 1.0) & (~self.used_ev_mask) )
        me_candidates = np.where(me_mask)[0]
        if (len(me_candidates) < 2):
            return None, None, None
    
        return se_iev, me_candidates[0], me_candidates[1]
    
    def process_data(self):
        
        # Use IO helper class to convert detector-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle', is_pp=False)
        self.df_fjparticles = io.load_data(m=self.m, offset_indices=True, min_pt=0.15)
        self.df_evts = io.track_df[['iev','centrality','z_vtx_reco']].drop_duplicates().set_index('iev', drop=False)
        self.nEvents = len(self.df_fjparticles.index)
        self.nTracks = len(io.track_df.index)
        # Pre-extract arrays for vectorized operations
        self.iev_array = self.df_evts['iev'].values
        self.vz_array = self.df_evts['z_vtx_reco'].values
        self.centrality_array = self.df_evts['centrality'].values

        print('Done with process_data()')


    def process_mc(self):
    
        # Use IO helper class to convert detector-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        io = process_io.ProcessIO(input_file=self.input_file_mc, track_tree_name='tree_Particle', use_ev_id_ext=False, is_det_level=True)
        # io = process_io.ProcessIO(input_file=self.input_file_mc, track_tree_name='tree_Particle_gen', use_ev_id_ext=False)
        self.df_fjparticles_mc = io.load_data(min_pt=0.15)
        self.df_evts_mc = io.track_df[['iev','z_vtx_reco']].drop_duplicates().set_index('iev', drop=False)
        # self.df_evts_mc = io.track_df[['iev','z_vtx_gen']].drop_duplicates().set_index('iev', drop=False)
        self.nEvents_mc = len(self.df_fjparticles_mc.index)
        self.nTracks_mc = len(io.track_df.index)
        self.pt_hat_bin = int(self.input_file_mc.split('/')[len(self.input_file_mc.split('/'))-4]) # depends on exact format of input_file name
        # print(self.df_evts_mc)

        print('Done with process_mc()')

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
    parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
    parser.add_argument('-fmc', '--inputFileMc', action='store',
                      type=str, metavar='inputFileMc',
                      default='/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC20g4/568/LHC20g4/7/296935/0002/AnalysisResults.root',
                      help='Path of ROOT file containing MC TTrees')

    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    process = ProcessEmbedENC(input_file=args.inputFile, config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.embed(args)