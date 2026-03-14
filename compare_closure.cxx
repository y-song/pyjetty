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

void FormatHist(TLegend *l, TH1 *hist, TString text, int markercolor = 1)
{
    hist->SetMarkerSize(0.5);
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(8);
    l->AddEntry(hist, text, "pl");
    
    hist->GetYaxis()->SetTitleOffset(1.25);
    hist->GetYaxis()->SetTitleSize(0.03); //(0.032);
    hist->GetYaxis()->SetLabelSize(0.03); //(0.032);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetXaxis()->SetTitleSize(0.03); //(0.032);
    hist->GetXaxis()->SetLabelSize(0.03); //(0.032);

    return;
}

void addLegendInfo(TLegend *l, string pt_min, string pt_max)
{
    l->SetTextSize(0.03);
    l->AddEntry("NULL", "PYTHIA8 jets + ALICE PbPb 0#minus10%", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "anti-#it{k}_{T} ch. jets, #it{R} = 0.2", "h");
    l->AddEntry("NULL", (pt_min + " < #it{p}_{T,jet}^{sub.} < " + pt_max + " GeV/#it{c}").c_str(), "h");
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

void compare_closure()
{
    SetStyle();

    const string jetR = "02";
    const string pt_min = "60";
    const string pt_max = "80";
    const string matched = "_matched";
    const string job_id_1 = "_48750700";
    const string job_id_2 = "_48695459";
    TLine *l = drawHoriLine(0.01, 0.2, 1.0, 1);

    // read in files
    TFile *f_eec_mbcone_file_1 = new TFile(("eec_mbcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + job_id_1 + ".root").c_str(), "READ");
    TFile *f_eec_mbcone_file_2 = new TFile(("eec_mbcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + job_id_2 + ".root").c_str(), "READ");

    // read in histograms
    TH1D *h_eec_ss_truth_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_ss" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    TH1D *h_eec_ss_mbcone_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_ss" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());  
    TH1D *h_eec_ss_truth_2 = (TH1D *)f_eec_mbcone_file_2->Get(("h_eec_ss" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    TH1D *h_eec_ss_mbcone_2 = (TH1D *)f_eec_mbcone_file_2->Get(("h_eec_ss" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());  
    // TH1D *h_eec_sb_truth_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_sb" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_sb_mbcone_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_sb" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_bb_truth_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_bb" + matched + "_truth_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());
    // TH1D *h_eec_bb_mbcone_1 = (TH1D *)f_eec_mbcone_file_1->Get(("h_eec_bb" + matched + "_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max).c_str());

    // divide over truth
    h_eec_ss_mbcone_1->Divide(h_eec_ss_truth_1);
    h_eec_ss_mbcone_2->Divide(h_eec_ss_truth_2);
    // h_eec_sb_mbcone_1->Divide(h_eec_sb_truth_1);
    // h_eec_bb_mbcone_1->Divide(h_eec_bb_truth_1);
    
    // set up histograms
    h_eec_ss_mbcone_1->GetXaxis()->SetRangeUser(0.01, 0.2);
    h_eec_ss_mbcone_2->GetXaxis()->SetRangeUser(0.01, 0.2);
    // h_eec_sb_mbcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    // h_eec_bb_mbcone->GetXaxis()->SetRangeUser(0.01, 0.2);
    h_eec_ss_mbcone_1->GetYaxis()->SetRangeUser(0.7, 1.5);
    h_eec_ss_mbcone_1->SetXTitle("#it{R}_{L}");
    h_eec_ss_mbcone_1->SetYTitle("(EEC, bkg. sub.) / (EEC, sig-sig)");
    
    // set up canvas
    TCanvas *c2 = new TCanvas();
    c2->SetCanvasSize(500, 500);
    c2->cd();
    gPad->SetLogx();

    // set up legend
    TLegend *leg = new TLegend(0.18, 0.6, 0.58, 0.8885185, "");
    addLegendInfo(leg, "60", "80");

    // draw histograms and lines
    FormatHist(leg, h_eec_ss_mbcone_1, "#rhoA subtraction", kBlue);
    FormatHist(leg, h_eec_ss_mbcone_2, "#splitline{Constituent subtraction +}{WTA recluster}", kGreen+2);
    h_eec_ss_mbcone_1->Draw();
    h_eec_ss_mbcone_2->Draw("same");
    // h_eec_sb_mbcone->Draw("same");
    // h_eec_bb_mbcone->Draw("same");
    // l4->Draw();
    // l5->Draw();
    l->Draw();
    leg->Draw("same");

    c2->SaveAs(("cfactor_mbcone" + matched + "_R" + jetR + "_" + pt_min + "_" + pt_max + job_id_1 + job_id_2 + ".pdf").c_str());

}