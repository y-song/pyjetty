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
}

void FormatHist(TLegend *l, TH1 *hist, TString text)
{
    hist->SetMarkerSize(1.0);
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

void compare_pt()
{
    SetStyle();

    const string jetR = "04";
    const string jetRPoint = "0.4";
    const string jobID = "49698333";
    const string option = "single_event_embed_asub_vs_csub";
    const double area_cut = 0.6*M_PI*std::stod(jetRPoint)*std::stod(jetRPoint);

    TFile *f = new TFile(("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/" + jobID + "/AnalysisResultsFinal.root").c_str(), "READ");
    TH2D *h1 = (TH2D *)f->Get(("h_asub_area_JetPt_combined_sub_all_R" + jetR).c_str());
    TH2D *h2 = (TH2D *)f->Get(("h_csub_area_JetPt_combined_sub_all_R" + jetR).c_str());
    TH2D *h1_clone = (TH2D *) h1->Clone("h1_clone");
    TH2D *h2_clone = (TH2D *) h2->Clone("h2_clone");
    h1_clone->GetYaxis()->SetRangeUser(area_cut, 1.0);
    h2_clone->GetYaxis()->SetRangeUser(area_cut, 1.0);
    TH1D *h1_proj = h1_clone->ProjectionX();
    TH1D *h2_proj = h2_clone->ProjectionX();
    // TH2D *h1 = (TH2D *)f->Get(("h_asub_JetPt_combined_sub_all_R" + jetR).c_str());
    // TH2D *h2 = (TH2D *)f->Get(("h_csub_JetPt_combined_sub_all_R" + jetR).c_str());

    cout << "Integral: " << h1_proj->Integral() << ", " << h2_proj->Integral() << endl;

    TCanvas *c1 = new TCanvas();
    c1->SetCanvasSize(700, 500);
    c1->cd();
    c1->SetLogy();
    
    TLegend *leg1 = new TLegend(0.55, 0.6, 0.8562155, 0.8885185, "");
    addLegendInfo(leg1, "", "", jetRPoint);
    h1_proj->SetLineColor(kGreen + 2);
    h2_proj->SetLineColor(kRed);
    h1_proj->GetXaxis()->SetTitle("p_{T}^{combined jet, sub} [GeV]");
    h1_proj->GetXaxis()->SetRangeUser(40, 200);
    h1_proj->GetYaxis()->SetTitle("");
    // h1_proj->GetYaxis()->SetRangeUser(1e-5, 0.3);
    h1_proj->SetTitle(option.c_str());
    FormatHist(leg1, h1_proj, "#rhoA subtract");
    FormatHist(leg1, h2_proj, "constituent subtract");

    h1_proj->Draw();
    h2_proj->Draw("same");
    leg1->Draw("same");

    c1->SaveAs(("pt_R" + jetR + "_" + option + "_" + jobID + ".pdf").c_str());

}
