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
    gStyle->SetOptTitle(1);
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
    gStyle->SetLabelOffset(0.01, "y");
    gStyle->SetLabelOffset(0.01, "x");
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
    gStyle->SetOptLogy();
}

void FormatHist(TLegend *l, TH1 *hist, TString text)
{
    hist->SetMarkerSize(0.5);
    hist->GetYaxis()->SetTitleOffset(1.05);
    hist->GetYaxis()->SetTitleSize(0.03); //(0.032);
    hist->GetYaxis()->SetLabelSize(0.03); //(0.032);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetXaxis()->SetTitleSize(0.03); //(0.032);
    hist->GetXaxis()->SetLabelSize(0.03); //(0.032);

    l->AddEntry(hist, text, "pl");

    return;
}

void addLegendInfo(TLegend *l, string pt_min, string pt_max, string jetR)
{
    l->SetTextSize(0.03);
    // l->AddEntry("NULL", "PYTHIA jets + thermal, no det. effects", "h");
    l->AddEntry("NULL", "PYTHIA8 jets + ALICE PbPb 0#minus10%", "h");
    // l->AddEntry("NULL", "ALICE PbPb 0#minus10%, #sqrt{#it{s}_{NN}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV, #hat{#it{p}}_{T} > 28 GeV", "h");
    l->AddEntry("NULL", ("charged jets, anti-#it{k}_{T}, #it{R} =" + jetR).c_str(), "h");
    l->AddEntry("NULL", "", "h");
    // l->AddEntry("NULL", "#it{p}_{T}^{combined, sub.} > 40 GeV", "h");
    // l->AddEntry("NULL", (pt_min + " < #it{p}_{T}^{combined, sub.} < " + pt_max + " GeV").c_str(), "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

void plot_dr()
{
    SetStyle();

    const string jetR = "02";
    const string jetRPoint = "0.2";
    const string jobID = "48875062";
    const string subtraction = "csub_wta";

    TFile *f = new TFile(("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/" + jobID + "/AnalysisResultsFinal.root").c_str(), "READ");
    // TH2D *h_dr_jetpt_combined_sub_matched = (TH2D *)f->Get(("h_dr_JetPt_combined_sub_matched_R" + jetR).c_str());
    // TFile *f = new TFile("AnalysisResults.root", "READ");
    TH2D *h = (TH2D *)f->Get(("h_" + subtraction + "_dr_JetPt_combined_sub_matched_R" + jetR).c_str());
    TH2D *h_pp = (TH2D *)f->Get(("h_" + subtraction + "_dr_JetPt_pp_matched_R" + jetR).c_str());

    TH2D *h1 = (TH2D *)h->Clone("h1");
    TH2D *h2 = (TH2D *)h->Clone("h2");
    TH2D *h3 = (TH2D *)h->Clone("h3");
    TH2D *h7 = (TH2D *)h_pp->Clone("h7");
    TH2D *h8 = (TH2D *)h_pp->Clone("h8");
    TH2D *h9 = (TH2D *)h_pp->Clone("h9");

    h1->GetXaxis()->SetRangeUser(40.0, 60.0);
    h2->GetXaxis()->SetRangeUser(60.0, 80.0);
    h3->GetXaxis()->SetRangeUser(80.0, 100.0);
    h7->GetXaxis()->SetRangeUser(40.0, 60.0);
    h8->GetXaxis()->SetRangeUser(60.0, 80.0);
    h9->GetXaxis()->SetRangeUser(80.0, 100.0);

    TH1D *h1_proj = h1->ProjectionY();
    TH1D *h2_proj = h2->ProjectionY();
    TH1D *h3_proj = h3->ProjectionY();
    TH1D *h7_proj = h7->ProjectionY();
    TH1D *h8_proj = h8->ProjectionY();
    TH1D *h9_proj = h9->ProjectionY();

    cout << "Mean: " << h7_proj->GetMean() << ", " << h8_proj->GetMean() << ", " << h9_proj->GetMean() << endl;
    cout << "Sigma: " << h7_proj->GetStdDev() << ", " << h8_proj->GetStdDev() << ", " << h9_proj->GetStdDev() << endl;

    h1_proj->Scale(1.0 / h1_proj->Integral());
    h2_proj->Scale(1.0 / h2_proj->Integral());
    h3_proj->Scale(1.0 / h3_proj->Integral());
    h7_proj->Scale(1.0 / h7_proj->Integral());
    h8_proj->Scale(1.0 / h8_proj->Integral());
    h9_proj->Scale(1.0 / h9_proj->Integral());

    // First canvas
    TCanvas *c1 = new TCanvas();
    c1->SetCanvasSize(700, 500);
    c1->cd();
    
    TLegend *leg1 = new TLegend(0.55, 0.6, 0.8562155, 0.8885185, "");
    addLegendInfo(leg1, "", "", jetRPoint);
    h1_proj->SetLineColor(kGreen + 2);
    h2_proj->SetLineColor(kRed);
    h1_proj->GetXaxis()->SetTitle("#DeltaR(combined sub, pp det)");
    h1_proj->GetYaxis()->SetTitle("Probability");
    h1_proj->GetXaxis()->SetRangeUser(0, 0.1);
    h1_proj->GetYaxis()->SetRangeUser(0.0001, 1.0);
    h1_proj->SetTitle(("Matched jets, " + subtraction).c_str());
    FormatHist(leg1, h1_proj, "40 < #it{p}_{T}^{combined sub} < 60 GeV");
    FormatHist(leg1, h2_proj, "60 < #it{p}_{T}^{combined sub} < 80 GeV");
    FormatHist(leg1, h3_proj, "80 < #it{p}_{T}^{combined sub} < 100 GeV");

    h1_proj->Draw();
    h2_proj->Draw("same");
    h3_proj->Draw("same");  
    leg1->Draw("same");

    c1->SaveAs(("dr_vs_combined_pt_matched_R" + jetR + "_" + subtraction + "_" + jobID + ".pdf").c_str());

    // Third canvas
    TCanvas *c3 = new TCanvas();
    c3->SetCanvasSize(700, 500);
    c3->cd();
    
    TLegend *leg3 = new TLegend(0.55, 0.6, 0.8562155, 0.8885185, "");
    addLegendInfo(leg3, "", "", jetRPoint);
    h7_proj->SetLineColor(kGreen + 2);
    h8_proj->SetLineColor(kRed);
    h7_proj->GetXaxis()->SetTitle("#DeltaR(combined sub, pp det)");
    h7_proj->GetYaxis()->SetTitle("Probability");
    h7_proj->GetXaxis()->SetRangeUser(0, 0.1);
    h7_proj->GetYaxis()->SetRangeUser(0.0001, 1.0);
    h7_proj->SetTitle(("Matched jets, " + subtraction).c_str());
    FormatHist(leg3, h7_proj, "40 < #it{p}_{T}^{pp det} < 60 GeV");
    FormatHist(leg3, h8_proj, "60 < #it{p}_{T}^{pp det} < 80 GeV");
    FormatHist(leg3, h9_proj, "80 < #it{p}_{T}^{pp det} < 100 GeV");

    h7_proj->Draw();
    h8_proj->Draw("same");
    h9_proj->Draw("same");  
    leg3->Draw("same");

    c3->SaveAs(("dr_vs_pp_pt_matched_R" + jetR + "_" + subtraction + "_" + jobID + ".pdf").c_str());
}
