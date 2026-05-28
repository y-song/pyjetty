#include <typeinfo>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <TStyle.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TLegend.h>
#include <TLine.h>
#include <TFile.h>

using namespace std;

void SetStyle(Bool_t graypalette = true)
{
    gStyle->Reset("Plain");
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    if (graypalette)
        gStyle->SetPalette(8, 0);
    else
        gStyle->SetPalette(1);
    gStyle->SetCanvasColor(10);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetFrameLineWidth(1);
    gStyle->SetFrameFillColor(kWhite);
    gStyle->SetPadColor(10);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetHistLineWidth(1);
    gStyle->SetHistLineColor(kRed);
    gStyle->SetFuncWidth(2);
    gStyle->SetFuncColor(kGreen);
    gStyle->SetLineWidth(1);
    gStyle->SetLabelSize(0.02, "xyz");
    gStyle->SetLabelOffset(0.01, "y"); //(0.005,"y");
    gStyle->SetLabelOffset(0.01, "x"); //(0.005,"x");
    gStyle->SetLabelColor(kBlack, "xyz");
    gStyle->SetTitleSize(0.025, "xyz");
    gStyle->SetTitleOffset(1.25, "y");
    gStyle->SetTitleOffset(1.2, "x");
    gStyle->SetTitleFillColor(kWhite);
    gStyle->SetTextSizePixels(26);
    gStyle->SetTextFont(42);
    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    gStyle->SetLegendFont(42);
}

void addLegendInfo(TLegend *l, string pt_min, string pt_max)
{
    l->SetTextSize(0.037);
    // l->AddEntry("NULL", "PYTHIA jets + thermal, no det. effects", "h");
    l->AddEntry("NULL", "PYTHIA8 jets + PbPb 0#minus10%", "h");
    // l->AddEntry("NULL", "ALICE PbPb 0#minus10%, #sqrt{#it{s}_{NN}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV, #hat{#it{p}}_{T} > 28 GeV", "h");
    l->AddEntry("NULL", "charged jets, anti-#it{k}_{T}, #it{R} = 0.4", "h");
    l->AddEntry("NULL", (pt_min + " < #it{p}_{T}^{combined jet, sub.} < " + pt_max + " GeV").c_str(), "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

TLine *drawHoriLine(double x1, double x2, double y1, int color, int linestyle = 2)
{
    auto fhoriline = new TLine(x1, y1, x2, y1);
    fhoriline->SetLineWidth(1);
    fhoriline->SetLineColor(color);
    fhoriline->SetLineStyle(linestyle);
    return fhoriline;
}

void plot_cfactor()
{
    SetStyle();

    const string jetR = "02";
    const string pt_min = "60";
    const string pt_max = "80";
    const string matched = "_matched";
    const string job_id = "_48750700";
    TLine *l = drawHoriLine(0.01, 0.2, 1.0, 1);

    // read in files
    // TFile *f_rho_file = new TFile(("rho_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + job_id + ".root").c_str(), "READ");
    // TFile *f_eec_perpcone_file = new TFile(("eec_perpcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + job_id + ".root").c_str(), "READ");
    TFile *f_eec_mbcone_file = new TFile(("eec_mbcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + job_id + ".root").c_str(), "READ");

    // read in histograms
    // TH1D *h_rho_jetcone = (TH1D *)f_rho_file->Get(("h_rho_jetcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_clone").c_str());
    // // TH1D *h_rho_perpcone = (TH1D *)f_rho_file->Get(("h_rho_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_clone").c_str());
    // TH1D *h_rho_mbcone = (TH1D *)f_rho_file->Get(("h_rho_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max + "_clone").c_str());

    TH1D *h_eec_ss_truth = (TH1D *)f_eec_mbcone_file->Get(("h_eec_ss" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_ss_perpcone = (TH1D *)f_eec_perpcone_file->Get(("h_eec_ss_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    TH1D *h_eec_ss_mbcone = (TH1D *)f_eec_mbcone_file->Get(("h_eec_ss" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    
    TH1D *h_eec_sb_truth = (TH1D *)f_eec_mbcone_file->Get(("h_eec_sb" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_sb_perpcone = (TH1D *)f_eec_perpcone_file->Get(("h_eec_sb_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    TH1D *h_eec_sb_mbcone = (TH1D *)f_eec_mbcone_file->Get(("h_eec_sb" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());

    TH1D *h_eec_bb_truth = (TH1D *)f_eec_mbcone_file->Get(("h_eec_bb" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_bb_perpcone = (TH1D *)f_eec_perpcone_file->Get(("h_eec_bb_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    TH1D *h_eec_bb_mbcone = (TH1D *)f_eec_mbcone_file->Get(("h_eec_bb" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());

    // double r_perpcone = h_rho_perpcone->GetMean()/h_rho_jetcone->GetMean();
    // cout << "r_perpcone: " << r_perpcone << endl;
    // double r_mbcone = h_rho_mbcone->GetMean()/h_rho_jetcone->GetMean();
    // cout << "r_mbcone: " << r_mbcone << endl;

    // divide over truth
    // h_eec_ss_perpcone->Divide(h_eec_ss_truth);
    // h_eec_sb_perpcone->Divide(h_eec_sb_truth);
    // h_eec_bb_perpcone->Divide(h_eec_bb_truth);
    h_eec_ss_mbcone->Divide(h_eec_ss_truth);
    h_eec_sb_mbcone->Divide(h_eec_sb_truth);
    h_eec_bb_mbcone->Divide(h_eec_bb_truth);
    
    // set up histograms
    // h_eec_ss_perpcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    // h_eec_sb_perpcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    // h_eec_bb_perpcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    // h_eec_sb_perpcone->GetYaxis()->SetRangeUser(0.4, 1.6);
    // h_eec_sb_perpcone->SetYTitle("perp cone / embed");
    // h_eec_sb_perpcone->SetLineColor(kGreen + 2);
    // h_eec_bb_perpcone->SetLineColor(kRed);
    h_eec_ss_mbcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    h_eec_sb_mbcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    h_eec_bb_mbcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    h_eec_ss_mbcone->GetYaxis()->SetRangeUser(0.5, 1.5);
    h_eec_ss_mbcone->GetYaxis()->SetTitleSize(0.04);
    h_eec_ss_mbcone->GetYaxis()->SetTitleOffset(1.5);
    h_eec_ss_mbcone->GetYaxis()->SetLabelSize(0.045);
    h_eec_ss_mbcone->SetYTitle("ME cone estimate / embed truth");
    h_eec_ss_mbcone->SetLineColor(kBlue);
    h_eec_sb_mbcone->SetLineColor(kGreen + 2);
    h_eec_bb_mbcone->SetLineColor(kRed);
    
    // set up legend
    TLegend *leg = new TLegend(0.40, 0.7, 0.8562155, 0.8885185, "");
    
    // make perpcone plot
    // set up lines
    // TLine *l1 = drawHoriLine(0.01, 0.2, r_perpcone, kGreen + 2);
    // TLine *l2 = drawHoriLine(0.01, 0.2, r_perpcone*r_perpcone, kRed);
    
    // set up canvas
    // TCanvas *c1 = new TCanvas();
    // c1->SetCanvasSize(500, 500);
    // c1->cd();
    // gPad->SetLogx();

    // // draw legend
    // addLegendInfo(leg, pt_min, pt_max);
    
    // draw histograms and lines
    // h_eec_ss_perpcone->Draw();
    // h_eec_sb_perpcone->Draw();
    // h_eec_bb_perpcone->Draw("same");
    // l1->Draw();
    // l2->Draw();
    // l->Draw();
    // leg->Draw("same");

    // c1->SaveAs(("cfactor_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max + job_id + ".pdf").c_str());

    // make mbcone plot
    // set up lines
    // TLine *l4 = drawHoriLine(0.01, 0.4, r_mbcone, kGreen + 2);
    // TLine *l5 = drawHoriLine(0.01, 0.4, r_mbcone*r_mbcone, kRed);
    
    // set up canvas
    TCanvas *c2 = new TCanvas();
    c2->SetCanvasSize(500, 500);
    c2->cd();
    gPad->SetLogx();

    // draw legend
    // addLegendInfo(leg, pt_min, pt_max);
    
    // draw histograms and lines
    // h_eec_ss_mbcone->GetYaxis()->SetRangeUser(0.95, 1.35);
    h_eec_ss_mbcone->Draw();
    h_eec_sb_mbcone->Draw("same");
    h_eec_bb_mbcone->Draw("same");
    // l4->Draw();
    // l5->Draw();
    l->Draw();
    // leg->Draw("same");

    c2->SaveAs(("cfactor_mbcone" + matched + "_R" + jetR + "_" + pt_min + "_" + pt_max + job_id + ".pdf").c_str());

}