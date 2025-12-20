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

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext # /global/cfs/cdirs/alice/youqi/mypyjetty/heppy/cpptools/src/pythiafjext/
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base
from pyjetty.mputils.csubtractor import CEventSubtractor
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

            R_label = str(jetR).replace('.', '')# + 'Scaled'

            name = 'h_JetPt_ch_combined_R{}'.format(R_label)
            pt_bins = linbins(0,1000,500)
            h = ROOT.TH1D(name, name, 500, pt_bins)
            h.GetYaxis().SetTitle('p_{T, comb jet}')
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)
            
            name = 'h_area_JetPt_ch_combined_R{}'.format(R_label)
            pt_bins = linbins(0,200,200)
            area_bins = linbins(0,1,100)
            h = ROOT.TH2D(name, name, 200, pt_bins, 100, area_bins)
            h.GetXaxis().SetTitle('p_{T, comb jet}')
            h.GetYaxis().SetTitle('Area')
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

                    for coneR in self.coneR_list:

                        # jetcone combined
                        name = 'h_jetcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)
                        
                        # wta cone combined
                        name = 'h_wta_jetcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)
                        
                        # mbcone combined
                        name = 'h_mbcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # wta mbcone combined
                        name = 'h_wta_mbcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)
                        
                        # perpcone combined
                        name = 'h_perpcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, obs_nbins, obs_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle(observable)
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # wta perpcone combined
                        name = 'h_wta_perpcone{}_{}_JetPt_ch_combined_R{}_{}'.format(coneR, observable, R_label, thrd_label)
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

                for pair_type_label in self.pair_type_labels:

                    # truth information
                    
                    name = 'h_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_jetcone_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)
                    
                    name = 'h_wta_jetcone_ENC{}_JetPt_ch_combined_R{}_{}'.format(str(ipoint)+pair_type_label, R_label, thrd_label)
                    print('Initialize histogram',name)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T, comb jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    for coneR in self.coneR_list:

                        # jet mbcone combined
                        name = 'h_jet_mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # jetcone mbcone combined
                        name = 'h_jetcone_mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # wta jetcone mbcone combined
                        name = 'h_wta_jetcone_mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # jet perpcone combined
                        name = 'h_jet_perpcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # jetcone perpcone combined
                        name = 'h_jetcone_perpcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        # wta jetcone perpcone combined
                        name = 'h_wta_jetcone_perpcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                        print('Initialize histogram',name)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('p_{T, comb jet}')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        if (pair_type_label == '_sb'):

                            name = 'h_2perpcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_wta_2perpcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_2mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
                            print('Initialize histogram',name)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('p_{T, pp jet}')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_wta_2mbcone{}_ENC{}_JetPt_ch_combined_R{}_{}'.format(coneR, str(ipoint)+pair_type_label, R_label, thrd_label)
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

            # copied from https://github.com/pdhankhe/pyjetty/blob/master/pyjetty/alihfjets/dev/hfjet/process/user/hf_jetaxes/process_data_hfjet_jetaxes_diff.py#L147
            jet_def_wta = fj.JetDefinition(fj.cambridge_algorithm, 2*jetR)
            jet_def_wta.set_recombination_scheme(fj.WTA_pt_scheme)
            setattr(self, "jet_def_wta_R%s" % jetR_str, jet_def_wta)
            print(jet_def_wta)
        
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
    def analyze_events(self):
        
        jetR_str = str(self.jetR_list[0]).replace('.', '')
        jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
        jet_def = getattr(self, "jet_def_R%s" % jetR_str)
        track_selector_ch = getattr(self, "track_selector_ch")
        jet_pt_thrd = self.check_jet_pt_thrd()

        print("Nevt(PbPb):", self.nEvents)
        print("Nevt(MC): ", self.nEvents_mc)
        
        iev_mc = 0 # Event loop count
        used_ev = []

        while (iev_mc < self.nEvents_mc):
            
            # assuming they are charged final states...
            self.parts_pythia_ch = fj.vectorPJ(self.df_fjparticles_mc.iloc[iev_mc])

            cs_pp = fj.ClusterSequence(track_selector_ch(self.parts_pythia_ch), jet_def)
            jets_pp = fj.sorted_by_pt( jet_selector(cs_pp.inclusive_jets()) )
            
            # if leading jet pT is over the thrd for the given pTHat bin, go to next MC
            if (len(jets_pp) > 0 and jets_pp[0].perp() > jet_pt_thrd):
                print("skip due to high weight")
                iev_mc += 1
                continue

            # check MC event info
            mc_vtx = self.df_evts_mc.iloc[iev_mc]["z_vtx_reco"]
            # print("MC ievt:", iev_mc, ", vtx =", mc_vtx)

            # require the SE to be within 1 cm of z_vtx of MC, and has not been used for ME
            df_evts_select_se = self.df_evts[(self.df_evts.z_vtx_reco < mc_vtx + 1) & (self.df_evts.z_vtx_reco > mc_vtx - 1)]
            
            # no acceptable SE, go to next MC
            if (df_evts_select_se.shape[0] == 0):
                iev_mc += 1
                continue
            
            df_evts_select_se_index = 0 # start with the 0th row of the selected SE df
            
            # check PbPb SE info
            se_iev = (int)(df_evts_select_se.iloc[df_evts_select_se_index]["iev"])
            se_centrality = self.df_evts.iloc[se_iev]["centrality"]
            se_vtx = self.df_evts.iloc[se_iev]["z_vtx_reco"]
            # print("SE ievt:", se_iev, ", vtx =", se_vtx, ", centrality =", se_centrality)

            # require the ME to be within 2% of centrality and 1 cm of z_vtx, and has not been used for ME
            df_evts_select = self.df_evts[(self.df_evts.centrality < se_centrality + 2) & (self.df_evts.centrality > se_centrality - 2) & (self.df_evts.z_vtx_reco < se_vtx + 1) & (self.df_evts.z_vtx_reco > se_vtx - 1)]
            
            # less than 2 acceptable ME
            while (df_evts_select.shape[0] < 2):
                
                # go to next SE, if there is more acceptable SE
                if (df_evts_select_se_index < df_evts_select_se.shape[0] - 1):
                    
                    df_evts_select_se_index += 1
                    
                    # check PbPb SE info
                    se_iev = (int)(df_evts_select_se.iloc[df_evts_select_se_index]["iev"])
                    se_centrality = self.df_evts.iloc[se_iev]["centrality"]
                    se_vtx = self.df_evts.iloc[se_iev]["z_vtx_reco"]
                    # print("SE ievt:", se_iev, ", vtx =", se_vtx, ", centrality =", se_centrality)

                    # require the ME to be within 2% of centrality and 1 cm of z_vtx, and has not been used for ME
                    df_evts_select = self.df_evts[(self.df_evts.centrality < se_centrality + 2) & (self.df_evts.centrality > se_centrality - 2) & (self.df_evts.z_vtx_reco < se_vtx + 1) & (self.df_evts.z_vtx_reco > se_vtx - 1)]

                # no acceptable SE, go to next MC
                else:
                    iev_mc += 1
                    break

            # no acceptable SE, go to next MC
            if (df_evts_select.shape[0] < 2):
                continue

            # check PbPb ME info
            me_iev = (int)(df_evts_select.iloc[0]["iev"])
            me_centrality = self.df_evts.iloc[me_iev]["centrality"]
            me_vtx = self.df_evts.iloc[me_iev]["z_vtx_reco"]
            # print("ME ievt:", me_iev, ", vtx =", me_vtx, ", centrality =", me_centrality)

            # check PbPb ME2 info
            me2_iev = (int)(df_evts_select.iloc[1]["iev"])
            me2_centrality = self.df_evts.iloc[me2_iev]["centrality"]
            me2_vtx = self.df_evts.iloc[me2_iev]["z_vtx_reco"]
            # print("ME ievt:", me2_iev, ", vtx =", me2_vtx, ", centrality =", me2_centrality)
            
            # read in a SE
            self.fj_particles_combined_beforeCS = self.get_mixed_event(se_iev)
            # read in a ME
            self.fj_particles_combined_beforeCS_mb1 = fj.vectorPJ(self.df_fjparticles.iloc[me_iev])
            used_ev.append(me_iev)
            # read in another ME
            self.fj_particles_combined_beforeCS_mb2 = fj.vectorPJ(self.df_fjparticles.iloc[me2_iev])
            used_ev.append(me2_iev)

            # Add particles from all pythia jets to the list
            self.parts_pythia_ch_jet = fj.vectorPJ()
            for ijet in range(0, len(jets_pp)):
                for p in jets_pp[ijet].constituents():
                    self.parts_pythia_ch_jet.push_back(p)
            [self.fj_particles_combined_beforeCS.push_back(p) for p in self.parts_pythia_ch_jet]

            if self.debug_level > 1:
                for p in self.fj_particles_combined_beforeCS:
                    print('particle info user_index',p.user_index(),'pt',p.perp(),'phi',p.phi(),'eta',p.eta(),)

            self.constituent_subtractor = CEventSubtractor(max_distance=self.max_distance, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR)
            self.constituent_subtractor.process_event(self.fj_particles_combined_beforeCS)
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
        jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
        cs_combined = fj.ClusterSequenceArea(track_selector_ch(self.fj_particles_combined_beforeCS), jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
        area_cut = 0.6*np.pi*self.jetR_list[0]*self.jetR_list[0]
        pt_sub_cut = 40

        jets_combined = fj.sorted_by_pt( jet_selector(cs_combined.inclusive_jets()) )        
        jets_combined_select = fj.vectorPJ()

        for i in range(0, len(jets_combined)):
            # print("ijet, area, pt, eta, phi:", i, ",", jets_combined[i].area(), ",", jets_combined[i].perp(), ",", jets_combined[i].eta(), ",", jets_combined[i].phi())
            pt_sub = jets_combined[i].perp()-self.rho*jets_combined[i].area()
            if (pt_sub > pt_sub_cut and jets_combined[i].area() > area_cut):
                jets_combined_select.push_back(jets_combined[i])
                
        if (len(jets_combined_select) == 0):
            return

        R_label = str(self.jetR_list[0]).replace('.', '')# + 'Scaled'
        
        #-------------------------------------------------------------
        # loop over all selected combined jets
        #-------------------------------------------------------------
        for jet_combined in jets_combined_select:

            jet_combined_wta = reclusterer_wta.result(jet_combined)
        
            hname = 'h_JetPt_ch_combined_R{}'.format(R_label)
            getattr(self, hname).Fill(jet_combined.perp()-self.rho*jet_combined.area())
            
            self.fill_jets(jet_combined, self.jetR_list[0])
            self.fill_jetcone(jet_combined, jet_combined_wta, self.jetR_list[0])
            self.fill_perpcone(jet_combined, jet_combined_wta, self.jetR_list[0], self.coneR_list[0])
            self.fill_2perpcone(jet_combined, jet_combined_wta, self.jetR_list[0], self.coneR_list[0])
            self.fill_mbcone(jet_combined, jet_combined_wta, self.jetR_list[0], self.coneR_list[0])
            self.fill_2mbcone(jet_combined, jet_combined_wta, self.jetR_list[0], self.coneR_list[0])

    #---------------------------------------------------------------
    # Fill perp cone for matched combined jets
    #---------------------------------------------------------------
    def fill_jets(self, jet_combined, jetR):

        R_label = str(jetR).replace('.', '')# + 'Scaled'

        hname = 'h_area_JetPt_ch_combined_R{}'.format(R_label)
        if self.debug_level > 0:
            print('area',jet_combined.area(),'rho',self.rho,'combined jet pt after subtraction',jet_combined.perp()-self.rho*jet_combined.area())
        getattr(self, hname).Fill(jet_combined.perp()-self.rho*jet_combined.area(), jet_combined.area())
        
        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_ENC_histograms(hname, jet_combined, None)

        # fill EEC for matched comb jet using comb jet (after rho subtraction) for jet pT
        hname = 'h_{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_rho_local_histograms(hname, jet_combined, jetR, None)
            
    #---------------------------------------------------------------
    # Fill perp cone for matched combined jets
    #---------------------------------------------------------------
    def fill_perpcone(self, jet_combined, jet_combined_wta, jetR, coneR):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        perpcone_R = coneR
        # NB1: only enable dynamic option when coneR = jetR
        # NB2: similar result using dynamic and static cone
        if self.static_perpcone == False and coneR == jetR:
            perpcone_R = math.sqrt(jet_combined.area()/np.pi)
        
        # Do perp cone for the E-scheme jet and E-scheme jet cone
        perp_jet1 = fj.PseudoJet()
        perp_jet1.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi() + np.pi/2, jet_combined.m())
        parts_in_perpcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet1, perpcone_R)
        parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -np.pi/2)
        
        for mode in ['jet','jetcone']:
            # 1. E-scheme jet
            if (mode == 'jet'):
                constituents = jet_combined.constituents()
                parts_in_jet = self.copy_parts(constituents) # NB: make a copy so that the original jet constituents will not be modifed
            # 2. E-scheme jet cone
            elif (mode == 'jetcone'):
                parts_in_jet = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined, perpcone_R)

            # use 999 and -999 to distinguish from previous used labeling numbers
            parts_in_cone1 = fj.vectorPJ()
            # fill parts from jet
            for part in parts_in_jet: # everything in the jet cone is "signal"
                part.set_user_index(999)
                parts_in_cone1.append(part)
            # fill parts from perp cone 1
            for part in parts_in_perpcone1:
                part.set_user_index(-999)
                parts_in_cone1.append(part)

            # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
            hname = 'h_{}_perpcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(mode, coneR, R_label)
            self.fill_ENC_histograms(hname, jet_combined, parts_in_cone1)
            
            if (mode == 'jetcone'):

                hname = 'h_perpcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
                self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_cone1)
        
        # Do perp cone for the WTA jet cone
        perp_jet3 = fj.PseudoJet()
        perp_jet3.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi() + np.pi/2, jet_combined_wta.m())
        parts_in_perpcone3 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet3, perpcone_R)
        parts_in_perpcone3 = self.rotate_parts(parts_in_perpcone3, -np.pi/2)
        
        for mode in ['jetcone']:
            # 3. WTA jet cone
            if (mode == 'jetcone'):
                parts_in_jet = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined_wta, perpcone_R)

            # use 999 and -999 to distinguish from previous used labeling numbers
            parts_in_cone3 = fj.vectorPJ()
            # fill parts from jet
            for part in parts_in_jet: # everything in the jet cone is "signal"
                part.set_user_index(999)
                parts_in_cone3.append(part)
            # fill parts from perp cone 3
            for part in parts_in_perpcone3:
                part.set_user_index(-999)
                parts_in_cone3.append(part)
            
            # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
            hname = 'h_wta_{}_perpcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(mode, coneR, R_label)
            self.fill_ENC_histograms(hname, jet_combined, parts_in_cone3)

            hname = 'h_wta_perpcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
            self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_cone3)
            
    def fill_2perpcone(self, jet_combined, jet_combined_wta, jetR, coneR):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        perpcone_R = coneR
        # NB1: only enable dynamic option when coneR = jetR
        # NB2: similar result using dynamic and static cone
        if self.static_perpcone == False and coneR == jetR:
            perpcone_R = math.sqrt(jet_combined.area()/np.pi)
        
        # Do perp cone for the E-scheme jet and E-scheme jet cone
        perp_jet1 = fj.PseudoJet()
        perp_jet1.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi() + np.pi/2, jet_combined.m())
        perp_jet2 = fj.PseudoJet()
        perp_jet2.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi() - np.pi/2, jet_combined.m())
        parts_in_perpcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet1, perpcone_R)
        parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -np.pi/2)
        parts_in_perpcone2 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet2, perpcone_R)        
        parts_in_perpcone2 = self.rotate_parts(parts_in_perpcone2, +np.pi/2)
        
        # label one perpcone as "sig" and the other as "bkg" so the perp1-perp2 and perp1(2)-perp1(2) correlations can be saved separately
        parts_in_cone = fj.vectorPJ()
        for part in parts_in_perpcone1:
            part.set_user_index(999)
            parts_in_cone.append(part)
        for part in parts_in_perpcone2:
            part.set_user_index(-999)
            parts_in_cone.append(part)

        hname = 'h_2perpcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_cone)
        
        # Do perp cone for the WTA jet cone
        perp_jet3 = fj.PseudoJet()
        perp_jet3.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi() + np.pi/2, jet_combined_wta.m())
        perp_jet4 = fj.PseudoJet()
        perp_jet4.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi() - np.pi/2, jet_combined_wta.m())
        parts_in_perpcone3 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet3, perpcone_R)
        parts_in_perpcone3 = self.rotate_parts(parts_in_perpcone3, -np.pi/2)
        parts_in_perpcone4 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, perp_jet4, perpcone_R)        
        parts_in_perpcone4 = self.rotate_parts(parts_in_perpcone4, +np.pi/2)
        
        parts_in_cone = fj.vectorPJ()
        for part in parts_in_perpcone3:
            part.set_user_index(999)
            parts_in_cone.append(part)
        for part in parts_in_perpcone4:
            part.set_user_index(-999)
            parts_in_cone.append(part)
            
        hname = 'h_wta_2perpcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_cone) # Use the original E-scheme jet pT

    #---------------------------------------------------------------
    # Fill mb cone for matched combined jets
    #---------------------------------------------------------------
    def fill_mbcone(self, jet_combined, jet_combined_wta, jetR, coneR):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        mbcone_R = coneR
        # NB1: only enable dynamic option when coneR = jetR
        # NB2: similar result using dynamic and static cone
        if self.static_perpcone == False and coneR == jetR:
            mbcone_R = math.sqrt(jet_combined.area()/np.pi)
        
        # Do mb cones for the E-scheme jet and E-scheme jet cone
        mb_jet1 = fj.PseudoJet()
        mb_jet1.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi(), jet_combined.m())
        parts_in_mbcone1 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb1, mb_jet1, mbcone_R)
        
        for mode in ['jet','jetcone']:
            # 1. E-scheme jet
            if (mode == 'jet'):
                constituents = jet_combined.constituents()
                parts_in_jet = self.copy_parts(constituents) # NB: make a copy so that the original jet constituents will not be modifed
            # 2. E-scheme jet cone
            elif (mode == 'jetcone'):
                parts_in_jet = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined, mbcone_R)

            # use 999 and -999 to distinguish from previous used labeling numbers
            parts_in_cone1 = fj.vectorPJ()
            # fill parts from jet
            for part in parts_in_jet: # everything in the jet cone is "signal"
                part.set_user_index(999)
                parts_in_cone1.append(part)
            # fill parts from mb cone 1
            for part in parts_in_mbcone1:
                part.set_user_index(-999)
                parts_in_cone1.append(part)

            # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
            hname = 'h_{}_mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(mode, coneR, R_label)
            self.fill_ENC_histograms(hname, jet_combined, parts_in_cone1)

            if (mode == 'jetcone'):

                hname = 'h_mbcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
                self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_cone1)

        # Do mb cone for the WTA jet cone
        mb_jet3 = fj.PseudoJet()
        mb_jet3.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi(), jet_combined_wta.m())
        # mb_jet4 = fj.PseudoJet()
        # mb_jet4.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi(), jet_combined_wta.m())
        parts_in_mbcone3 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb1, mb_jet3, mbcone_R)
        # parts_in_mbcone4 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb2, mb_jet4, mbcone_R)        
        
        for mode in ['jetcone']:
            # 3. WTA jet cone
            if (mode == 'jetcone'):
                parts_in_jet = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined_wta, mbcone_R)

            # use 999 and -999 to distinguish from previous used labeling numbers
            parts_in_cone3 = fj.vectorPJ()
            # fill parts from jet
            for part in parts_in_jet: # everything in the jet cone is "signal"
                part.set_user_index(999)
                parts_in_cone3.append(part)
            # fill parts from mb cone 3
            for part in parts_in_mbcone3:
                part.set_user_index(-999)
                parts_in_cone3.append(part)
            
            # fill EEC for matched comb jet using comb jet (rho subtracted) for jet pT
            hname = 'h_wta_{}_mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(mode, coneR, R_label)
            self.fill_ENC_histograms(hname, jet_combined, parts_in_cone3)

            hname = 'h_wta_mbcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
            self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_cone3)

    def fill_2mbcone(self, jet_combined, jet_combined_wta, jetR, coneR):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'
        mbcone_R = coneR
        # NB1: only enable dynamic option when coneR = jetR
        # NB2: similar result using dynamic and static cone
        if self.static_perpcone == False and coneR == jetR:
            mbcone_R = math.sqrt(jet_combined.area()/np.pi)
        
        # Do perp cone for the E-scheme jet and E-scheme jet cone
        mb_jet1 = fj.PseudoJet()
        mb_jet1.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi(), jet_combined.m())
        mb_jet2 = fj.PseudoJet()
        mb_jet2.reset_PtYPhiM(jet_combined.pt(), jet_combined.rapidity(), jet_combined.phi(), jet_combined.m())
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
            
        hname = 'h_2mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_cone)
        
        # Do perp cone for the WTA jet cone
        mb_jet3 = fj.PseudoJet()
        mb_jet3.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi(), jet_combined_wta.m())
        mb_jet4 = fj.PseudoJet()
        mb_jet4.reset_PtYPhiM(jet_combined_wta.pt(), jet_combined_wta.rapidity(), jet_combined_wta.phi(), jet_combined_wta.m())
        parts_in_mbcone3 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb1, mb_jet3, mbcone_R)
        parts_in_mbcone4 = self.find_parts_around_jet(self.fj_particles_combined_beforeCS_mb2, mb_jet4, mbcone_R) 
        
        parts_in_cone = fj.vectorPJ()
        for part in parts_in_mbcone3:
            part.set_user_index(999)
            parts_in_cone.append(part)
        for part in parts_in_mbcone4:
            part.set_user_index(-999)
            parts_in_cone.append(part)

        hname = 'h_wta_2mbcone{}_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_cone) # Use the original E-scheme jet pT
    
    #---------------------------------------------------------------
    # Fill jet cone for matched combined jets
    #---------------------------------------------------------------
    def fill_jetcone(self, jet_combined, jet_combined_wta, jetR):

        R_label = str(jetR).replace('.', '') #+ 'Scaled'

        jetcone_R = jetR
        coneR = jetR

        parts_in_jetcone = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined, jetcone_R)
        parts_in_wta_jetcone = self.find_parts_around_jet(self.fj_particles_combined_beforeCS, jet_combined_wta, jetcone_R)

        hname = 'h_jetcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_jetcone)

        hname = 'h_jetcone_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_jetcone)

        hname = 'h_wta_jetcone{}_{{}}_JetPt_ch_combined_R{}_{{}}'.format(coneR, R_label)
        self.fill_rho_local_histograms(hname, jet_combined, coneR, parts_in_wta_jetcone)

        hname = 'h_wta_jetcone_ENC{{}}_JetPt_ch_combined_R{}_{{}}'.format(R_label)
        self.fill_ENC_histograms(hname, jet_combined, parts_in_wta_jetcone)

    #---------------------------------------------------------------
    # Fill matched ENC histograms
    #---------------------------------------------------------------
    def fill_ENC_histograms(self, hname, jet_combined, cone_parts):
        
        if cone_parts == None:
            constituents = fj.sorted_by_pt(jet_combined.constituents())
            # print("In fill_ENC_histograms...len(jet_combined.constituents()):", len(constituents))
        else:
            constituents = fj.sorted_by_pt(cone_parts)
            # print("In fill_ENC_histograms...len(cone_parts):", len(constituents))

        for thrd in self.thrd_list:
            c_select = fj.vectorPJ()
            thrd_label = 'trk{:.0f}'.format(thrd*10)
            for c in constituents:
                if c.pt() < thrd:
                    break
                c_select.append(c) # NB: use the break statement since constituents are already sorted
            # print("N(constituents) w/ thrd:", len(c_select))

            if 'combined' in hname:
                jet_pt_weight = jet_combined.perp()-self.rho*jet_combined.area()
                jet_pt_select = jet_combined.perp()-self.rho*jet_combined.area()

            new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt_weight, self.npoint, self.npower, self.dphi_cut, self.deta_cut) # NB: using the pp jet as reference for energy weight

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
    def fill_rho_local_histograms(self, hname, jet_combined, coneR, cone_parts):
        
        if cone_parts == None:
            constituents = fj.sorted_by_pt(jet_combined.constituents())
        else:
            constituents = fj.sorted_by_pt(cone_parts)

        if 'combined' in hname:
            jet_pt_select = jet_combined.perp()-self.rho*jet_combined.area()

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

            if 'jetcone' in hname or 'perpcone' in hname or 'mbcone' in hname:
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

    def process_data(self):
        
        # Use IO helper class to convert detector-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle', is_pp=False)
        self.df_fjparticles = io.load_data(m=self.m, offset_indices=True)
        self.df_evts = io.track_df[['iev','centrality','z_vtx_reco']].drop_duplicates().set_index('iev', drop=False)
        self.nEvents = len(self.df_fjparticles.index)
        self.nTracks = len(io.track_df.index)
        # Pre-extract arrays for vectorized operations
        self.vz_array = self.df_evts['z_vtx_reco'].values
        self.centrality_array = self.df_evts['centrality'].values

        print('Done with process_data()')

    def get_mixed_event(self, se_iev):

        ntrks_target = (int)(self.hmult.GetRandom())
        ntrks_added = 0
        iev_unique = []
        event_candidate_iev = 0
        
        mixed_event = fj.vectorPJ()
        mixed_event_vz = self.df_evts.iloc[se_iev]['z_vtx_reco']
        mixed_event_centrality = self.df_evts.iloc[se_iev]['centrality']

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
            if (event_candidate_iev not in iev_unique):
                iev_unique.append(event_candidate_iev)
            # print("itrk, event_candidate_iev:", ntrks_added, event_candidate_iev)

        if (ntrks_target % 100 == 0): # randomly check this
            print("ntrks, nevts_unique:", ntrks_target, len(iev_unique))

        return mixed_event

    def process_mc(self):
    
        # Use IO helper class to convert detector-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        io = process_io.ProcessIO(input_file=self.input_file_mc, track_tree_name='tree_Particle', use_ev_id_ext=False, is_det_level=True)
        # io = process_io.ProcessIO(input_file=self.input_file_mc, track_tree_name='tree_Particle_gen', use_ev_id_ext=False)
        self.df_fjparticles_mc = io.load_data(m=self.m)
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