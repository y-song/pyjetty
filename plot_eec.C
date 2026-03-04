#include <typeinfo>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <TStyle.h> // location: /global/cfs/cdirs/alice/heppy_soft/15-09-2024/yasp/software/root/default/include/
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
    gStyle->SetLabelSize(0.045, "xyz");
    gStyle->SetLabelOffset(0.01, "y"); //(0.005,"y");
    gStyle->SetLabelOffset(0.01, "x"); //(0.005,"x");
    gStyle->SetLabelColor(kBlack, "xyz");
    gStyle->SetTitleSize(0.05, "xyz");
    gStyle->SetTitleOffset(1.25, "y");
    gStyle->SetTitleOffset(1.2, "x");
    gStyle->SetTitleFillColor(kWhite);
    gStyle->SetTextSizePixels(26);
    gStyle->SetTextFont(42);
    // gStyle->SetTickLength(0.04,"X");  gStyle->SetTickLength(0.04,"Y");

    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(kWhite);
    // gStyle->SetFillColor(kWhite);
    gStyle->SetLegendFont(42);
}

void ProcessCanvas(TCanvas *Canvas)
{
    gStyle->SetOptStat(0);
    Canvas->SetHighLightColor(1);
    Canvas->SetFillColor(0);
    Canvas->SetBorderMode(0);
    Canvas->SetBorderSize(2);
    Canvas->SetTickx(1);
    Canvas->SetTicky(1);
    Canvas->SetFrameBorderMode(0);
    Canvas->SetFrameLineWidth(1);
    Canvas->SetFrameBorderMode(1);
}

void FormatHist(TLegend *l, TH1 *hist, std::string text, int markercolor = 1, int markerstyle = 8)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(0.5);
    l->AddEntry(hist, text.c_str(), "pl");
    
    hist->GetYaxis()->SetTitleOffset(1.05);
    hist->GetYaxis()->SetTitleSize(0.042);
    hist->GetYaxis()->SetLabelSize(0.042);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetXaxis()->SetTitleSize(0.042);
    hist->GetXaxis()->SetLabelSize(0.042);

    return;
}

TLine *drawVertLine(double x1, double y1, double y2, int color, int linestyle = 2)
{
    auto fvertline = new TLine(x1, y1, x1, y2);
    fvertline->SetLineWidth(1);
    fvertline->SetLineColor(color);
    fvertline->SetLineStyle(linestyle);
    return fvertline;
}

TLine *drawHoriLine(double x1, double x2, double y1, int color, int linestyle = 2)
{
    auto fhoriline = new TLine(x1, y1, x2, y1);
    fhoriline->SetLineWidth(1);
    fhoriline->SetLineColor(color);
    fhoriline->SetLineStyle(linestyle);
    return fhoriline;
}

void addLegendInfo(TLegend *l, TString jetR)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "PYTHIA8   pp #sqrt{#it{s}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "Anti-#it{k}_{T} charged-particle jets, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", "#it{p}_{T}^{track} > 1 GeV/#it{c}", "h");
    l->SetTextSize(0.037);
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

TH1D *DivideByBinWidth(TH1D *input_hist)
{
    TH1D *output_hist = (TH1D *)input_hist->Clone(Form("%s_clone", input_hist->GetName()));
    for (int ibin = 1; ibin < input_hist->GetNbinsX() + 1; ibin++)
    {
        double bincontent = input_hist->GetBinContent(ibin);
        double binerror = input_hist->GetBinError(ibin);
        double binwidth = input_hist->GetBinWidth(ibin);
        output_hist->SetBinContent(ibin, bincontent / binwidth);
        output_hist->SetBinError(ibin, binerror / binwidth);
    }
    return output_hist;
}

void plot_eec(std::string pt_min, std::string pt_max)
{

    gStyle->SetOptStat(0);
    SetStyle();

    std::string infile = "/global/cfs/cdirs/alice/youqi/mypyjetty/AnalysisResults.root";
    std::string jetR = "04";
    std::string jetR_str = "0.4";
    std::string outfile = "eec_R" + jetR + "_" + pt_min + "_" + pt_max + ".root";
   
    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << std::endl;
    std::cout << "R = " << jetR_str << std::endl;
    
    // initialize the input and output files
    TFile *f = new TFile(infile.c_str(), "READ");
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    // read in histograms from the input file
    TH1 *h_JetPt = (TH1 *)f->Get(("h_JetPt_R" + jetR).c_str());
    TH2 *h_EEC_JetPt = (TH2 *)f->Get(("h_EEC_JetPt_R" + jetR).c_str());

    // clone and process histograms
    TH2 *h_EEC_JetPt_clone = (TH2 *)h_EEC_JetPt->Clone(Form("%s_clone", h_EEC_JetPt->GetName()));
    h_EEC_JetPt_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
    TH1D *h_EEC_JetPt_clone_proj = h_EEC_JetPt_clone->ProjectionY();
    TH1D* h_EEC = DivideByBinWidth((TH1D *)h_EEC_JetPt_clone_proj->Clone(h_EEC_JetPt_clone_proj->GetName()));

    // normalize by the number of jets
    std::cout << "Number of jets from pT bins " << stof(pt_min) + 1 << "-" << stof(pt_max) << ": ";
    double njets = h_JetPt->Integral((int)(stof(pt_min) + 1), (int)(stof(pt_max)));
    std::cout << njets << std::endl;
    h_EEC->Scale(1.0 / njets);

    // make a canvas
    TCanvas *c = new TCanvas();
    ProcessCanvas(c);
    c->cd();
    gPad->SetLogx();

    // set legend
    TLegend *l = new TLegend(0.17, 0.57, 0.5, 0.88);
    std::string ptbin = pt_min + " < #it{p}_{T}^{jet} < " + pt_max + " GeV/#it{c}";
    addLegendInfo(l, jetR_str);

    // set histogram
    FormatHist(l, h_EEC, ptbin, kBlue, kFullCircle);
    h_EEC->GetXaxis()->SetRangeUser(0.01, 0.4);
    h_EEC->GetYaxis()->SetRangeUser(0, 8);
    h_EEC->GetXaxis()->SetTitle("#it{R}_{L}");
    h_EEC->GetYaxis()->SetTitle("#Sigma_{EEC}(#it{R}_{L})");

    // draw histogram and legend
    h_EEC->Draw("L");
    l->Draw("same");

    // save canvas to pdf
    std::string fname = "eec_R" + jetR + "_" + pt_min + "_" + pt_max + ".pdf";
    c->SaveAs(fname.c_str());
    delete c;
    delete l;

    // save histograms to output file
    f_out->cd();
    h_EEC->Write();
    f->Close();
    delete f;

    return;
}