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
import gc

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext # /global/cfs/cdirs/alice/youqi/mypyjetty/heppy/cpptools/src/pythiafjext/
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base
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
class ProcessEmbed(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(ProcessEmbed, self).__init__(
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

        self.jet_pt_thrd_combined = config["jet_pt_thrd_combined"]

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

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            name = 'hNevents_scaled'
            h = ROOT.TH1D(name, name, 2, -0.5, 1.5)
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)
            
            R_label = str(jetR).replace('.', '')

            for hist_prefix in ['asub_', 'csub_', 'csub_wta_']:
                
                name = 'h_{}JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,1000,500)
                h = ROOT.TH1D(name, name, 500, pt_bins)
                h.GetYaxis().SetTitle('p_{T, pp det}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,1000,500)
                h = ROOT.TH1D(name, name, 500, pt_bins)
                h.GetYaxis().SetTitle('p_{T, combined sub}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)
                
                name = 'h_{}JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,1000,500)
                h = ROOT.TH1D(name, name, 500, pt_bins)
                h.GetYaxis().SetTitle('p_{T, combined sub}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}area_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                area_bins = linbins(0,1,100)
                h = ROOT.TH2D(name, name, 200, pt_bins, 100, area_bins)
                h.GetXaxis().SetTitle('p_{T, pp det}')
                h.GetYaxis().SetTitle('Area_{combined sub}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}area_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                area_bins = linbins(0,1,100)
                h = ROOT.TH2D(name, name, 200, pt_bins, 100, area_bins)
                h.GetXaxis().SetTitle('p_{T, combined sub}')
                h.GetYaxis().SetTitle('Area_{combined sub}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}area_JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                area_bins = linbins(0,1,100)
                h = ROOT.TH2D(name, name, 200, pt_bins, 100, area_bins)
                h.GetXaxis().SetTitle('p_{T, combined sub}')
                h.GetYaxis().SetTitle('Area_{combined sub}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}dpt_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                dpt_bins = linbins(-100, 100, 200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, dpt_bins)
                h.GetXaxis().SetTitle('p_{T, pp det}')
                h.GetYaxis().SetTitle('#Deltap_{T} = p_{T, combined sub} - p_{T, pp det}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)
                
                name = 'h_{}dpt_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                dpt_bins = linbins(-100, 100, 200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, dpt_bins)
                h.GetXaxis().SetTitle('p_{T, combined sub}')
                h.GetYaxis().SetTitle('#Deltap_{T} = p_{T, combined sub} - p_{T, pp det}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}dpt_JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                dpt_bins = linbins(-100, 100, 200)
                h = ROOT.TH2D(name, name, 200, pt_bins, 200, dpt_bins)
                h.GetXaxis().SetTitle('p_{T, combined sub}')
                h.GetYaxis().SetTitle('#Deltap_{T} = p_{T, combined sub} - p_{T, pp det}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)
                
                name = 'h_{}dr_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                dr_bins = linbins(-0.05, 0.8, 170)
                h = ROOT.TH2D(name, name, 200, pt_bins, 170, dr_bins)
                h.GetXaxis().SetTitle('p_{T, pp det}')
                h.GetYaxis().SetTitle('#DeltaR(combined sub#minuspp det)')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_{}dr_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                pt_bins = linbins(0,200,200)
                dr_bins = linbins(-0.05, 0.8, 170)
                h = ROOT.TH2D(name, name, 200, pt_bins, 170, dr_bins)
                h.GetXaxis().SetTitle('p_{T, pp det}')
                h.GetYaxis().SetTitle('#DeltaR(combined sub#minuspp det)')
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

            jet_selector_high_pt = fj.SelectorPtMin(self.jet_pt_thrd_combined) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR)
            setattr(self, "jet_selector_high_pt_R%s" % jetR_str, jet_selector_high_pt)

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
            
            if (len(self.jets_pp) == 0):
                iev_mc += 1
                continue
            
            # if leading jet pT is over the thrd for the given pTHat bin, go to next MC
            if ( (len(self.jets_pp) > 0) and (self.jets_pp[0].perp() > jet_pt_thrd) ):
                print("skip due to high weight")
                iev_mc += 1
                continue

            # check MC event info
            mc_vtx = self.df_evts_mc.iloc[iev_mc]["z_vtx_reco"]
            se_iev = self.get_se(mc_vtx)

            if (se_iev == None):
                iev_mc += 1
                continue
            
            # read in a SE
            self.fj_particles_combined_beforeCS = fj.vectorPJ(self.df_fjparticles.iloc[se_iev])
            self.used_ev_mask[se_iev] = True

            # Add particles from all pythia jets to the list
            self.parts_pythia_ch_jet = fj.vectorPJ()
            for ijet in range(0, len(self.jets_pp)):
                for p in self.jets_pp[ijet].constituents():
                    self.parts_pythia_ch_jet.push_back(p)
            [self.fj_particles_combined_beforeCS.push_back(p) for p in self.parts_pythia_ch_jet]

            if (self.do_constituent_subtraction):
                self.constituent_subtractor = CEventSubtractor(max_distance=self.max_distance, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR)
            elif (self.do_ics):
                self.constituent_subtractor = ICEventSubtractor(max_distances=fjcontrib.vectorDouble(self.max_distances), alphas=fjcontrib.vectorDouble(self.alphas), max_eta=self.max_eta, jet_median_selector_R=self.jet_median_selector_R, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area)
            self.fj_particles_combined_afterCS = self.constituent_subtractor.process_event(self.fj_particles_combined_beforeCS)
            self.rho = self.constituent_subtractor.bge_rho.rho()

            self.hNevents.Fill(0)
            getattr(self, 'hNevents_scaled').Fill(0)
            self.analyze_jets()

            iev_mc += 1

    #---------------------------------------------------------------
    # Take pp jets, and embed them into PbPb
    #---------------------------------------------------------------
    def analyze_jets(self):
            
        jetR_str = str(self.jetR_list[0]).replace('.', '')
        jet_def = getattr(self, "jet_def_R%s" % jetR_str)
        track_selector_ch = getattr(self, "track_selector_ch")
        jet_selector_high_pt = getattr(self, "jet_selector_high_pt_R%s" % jetR_str)
        area_cut = 0 # 0.6*np.pi*self.jetR_list[0]*self.jetR_list[0]
        cs_combined_asub = None
        cs_combined_csub = None

        if (True): # area subtracted jets
            cs_combined_asub = fj.ClusterSequenceArea(track_selector_ch(self.fj_particles_combined_beforeCS), jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
            jets_combined_preselect = fj.sorted_by_pt( jet_selector_high_pt(cs_combined_asub.inclusive_jets()) )
            jets_combined_asub = []
            for jet_combined in jets_combined_preselect:
                pt_sub = jet_combined.perp()-self.rho*jet_combined.area()
                if (pt_sub < self.jet_pt_thrd_combined or jet_combined.area() < area_cut):
                    continue
                jets_combined_asub.append(jet_combined)
        
        if (True): # cs subtracted jets
            cs_combined_csub = fj.ClusterSequenceArea(track_selector_ch(self.fj_particles_combined_afterCS), jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
            jets_combined_preselect = fj.sorted_by_pt( jet_selector_high_pt(cs_combined_csub.inclusive_jets()) )
            jets_combined_csub = []
            for jet_combined in jets_combined_preselect:
                if (jet_combined.area() < area_cut):
                    continue
                jets_combined_csub.append(jet_combined)
                
        if (len(jets_combined_asub) == 0 and len(jets_combined_csub) == 0):
            return
    
        self.analyze_pp_matched_jets(cs_combined_asub, jets_combined_asub, "asub_")
        self.analyze_pp_matched_jets(cs_combined_csub, jets_combined_csub, "csub_")

    def analyze_pp_matched_jets(self, cluster_seq, jets_combined, hist_prefix=""):
        
        jetR_str = str(self.jetR_list[0]).replace('.', '')
        jet_def_wta = getattr(self, "jet_def_wta_R%s" % jetR_str)
        reclusterer_wta = fjcontrib.Recluster(jet_def_wta)
        R_label = str(self.jetR_list[0]).replace('.', '')
        
        # match pp jets to subtracted combined jets
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
        
        #-------------------------------------------------------------
        # loop over all selected combined jets
        #-------------------------------------------------------------
        for ijet_combined in range(0, len(jets_combined)):

            ijet_pp = jets_combined_matched_index[ijet_combined]
            pt_pp = 0.
            pt_combined_sub = jets_combined[ijet_combined].perp()
            area_combined_sub = jets_combined[ijet_combined].area()

            if ("asub" in hist_prefix):
                pt_combined_sub -= self.rho*jets_combined[ijet_combined].area()

            if (ijet_pp != -1): # combined jet has a matched pp counterpart

                pt_pp = self.jets_pp[ijet_pp].perp()
                dr = self.jets_pp[ijet_pp].delta_R(jets_combined[ijet_combined])
                
                hname = 'h_{}JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_pp)

                hname = 'h_{}JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_combined_sub)

                hname = 'h_{}dpt_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_pp, pt_combined_sub-pt_pp)

                hname = 'h_{}dpt_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_combined_sub, pt_combined_sub-pt_pp)

                hname = 'h_{}dr_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_pp, dr)

                hname = 'h_{}dr_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_combined_sub, dr)

                hname = 'h_{}area_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_pp, area_combined_sub)

                hname = 'h_{}area_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                getattr(self, hname).Fill(pt_combined_sub, area_combined_sub)

                if ("csub" in hist_prefix):

                    jet_pp_wta = reclusterer_wta.result(self.jets_pp[ijet_pp])

                    jet_combined_wta = reclusterer_wta.result(jets_combined[ijet_combined])
                    dr_wta = jet_pp_wta.delta_R(jet_combined_wta)
                    
                    hname = 'h_{}wta_dr_JetPt_pp_matched_R{}'.format(hist_prefix, R_label)
                    getattr(self, hname).Fill(pt_pp, dr_wta)

                    hname = 'h_{}wta_dr_JetPt_combined_sub_matched_R{}'.format(hist_prefix, R_label)
                    getattr(self, hname).Fill(pt_combined_sub, dr_wta)

            hname = 'h_{}JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
            getattr(self, hname).Fill(pt_combined_sub)

            hname = 'h_{}dpt_JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
            getattr(self, hname).Fill(pt_combined_sub, pt_combined_sub-pt_pp)

            hname = 'h_{}area_JetPt_combined_sub_all_R{}'.format(hist_prefix, R_label)
            getattr(self, hname).Fill(pt_combined_sub, area_combined_sub)
    
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

    def get_se(self, mc_vtx):
        
        se_mask = ( (np.abs(self.vz_array - mc_vtx) < 1.0) & (~self.used_ev_mask) )
        se_candidates = np.where(se_mask)[0]
        if (len(se_candidates) == 0):
            return None
        
        se_iev = se_candidates[0]
    
        return se_iev

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

    process = ProcessEmbed(input_file=args.inputFile, config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.embed(args)