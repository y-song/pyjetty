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
    std::cout << "Setting style!" << std::endl;

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

void FormatHist(TLegend *l, TH1 *hist, TString text, int markercolor = 1, int markerstyle = 8)
{
    hist->SetLineColor(markercolor);
    hist->SetMarkerColor(markercolor);
    hist->SetMarkerStyle(markerstyle);
    hist->SetMarkerSize(0.5);
    l->AddEntry(hist, text, "pl");

    // gPad->SetTickx();
    // gPad->SetTicky();
    //  h->SetLineWidth(2);
    hist->GetYaxis()->SetTitleOffset(1.05);
    hist->GetYaxis()->SetTitleSize(0.06); //(0.042);
    hist->GetYaxis()->SetLabelSize(0.05); //(0.042);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetXaxis()->SetTitleSize(0.06); //(0.042);
    hist->GetXaxis()->SetLabelSize(0.05); //(0.042);

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

void addLegendInfo(TLegend *l, TString ptbin, TString jetRPoint)
{
    l->SetTextSize(0.037);
    // l->AddEntry("NULL", "PYTHIA jets + thermal, no det. effects", "h");
    l->AddEntry("NULL", "PYTHIA8 jets + ALICE PbPb 0#minus10%", "h");
    // l->AddEntry("NULL", "ALICE PbPb 0#minus10%, #sqrt{#it{s}_{NN}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV, #hat{#it{p}}_{T} > 28 GeV", "h");
    l->AddEntry("NULL", "charged jets, anti-#it{k}_{T}, #it{R} = " + jetRPoint, "h");
    l->AddEntry("NULL", ptbin, "h");
    l->SetBorderSize(0);
    l->SetFillStyle(0); // turn legend transparent
}

TH1 *DivideByBinWidth(TH1 *input_hist)
{
    TH1 *output_hist = (TH1 *)input_hist->Clone(Form("%s_clone", input_hist->GetName()));
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

void plot_rho(string pt_min, string pt_max)
{

    gStyle->SetOptStat(0);
    SetStyle();

    // string one("");
    // const char infile[] = "/global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/AnalysisResults.root";
    const char infile[] = "/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/46833256/AnalysisResultsFinal.root";
    const string jetR = "02";
    const string jetRPoint = "0.2";
    TFile *f = new TFile(TString(infile), "READ");
    std::string add_name = "_46833256";
    std::cout << "output name will be " << add_name << std::endl;
    std::string outdir = "";
    std::string outfile = outdir + "rho_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    const std::string hist_names[] = {"jetcone"+jetRPoint+"_", "perpcone"+jetRPoint+"_", "mbcone"+jetRPoint+"_"};//, "wta_jetcone"+jetRPoint+"_"};
    std::cout << "checkpoint1" << std::endl;

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << endl;

    std::cout << "R = " << jetR << endl;

    const std::string h1_name = "h_" + hist_names[0] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";
    const std::string h2_name = "h_" + hist_names[1] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";
    const std::string h3_name = "h_" + hist_names[2] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";
    // const std::string h4_name = "h_" + hist_names[3] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";

    std::cout << "h1: " << h1_name.c_str() << endl;
    std::cout << "h2: " << h2_name.c_str() << endl;

    std::cout << "checkpoint2" << endl;

    // define pt related variables
    TString ptbin = pt_min + " < #it{p}_{T}^{combined jet, sub.} < " + pt_max + " GeV"; //;TString::Format("%s < #it{p}_{T}^{ch. jet} < %s GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5", pt_min, pt_max);
    std::string pdf_outdir = "";

    TH2 *h1_clone = (TH2 *)f->Get(h1_name.c_str())->Clone(Form("%s_clone", h1_name.c_str()));
    TH2 *h2_clone = (TH2 *)f->Get(h2_name.c_str())->Clone(Form("%s_clone", h2_name.c_str()));
    TH2 *h3_clone = (TH2 *)f->Get(h3_name.c_str())->Clone(Form("%s_clone", h3_name.c_str()));
    // TH2 *h4_clone = (TH2 *)f->Get(h4_name.c_str())->Clone(Form("%s_clone", h4_name.c_str()));

    h1_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
    h2_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
    h3_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
    // h4_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));

    std::cout << "checkpoint4" << endl;

    // Project onto observable axis
    TH1D *h1_proj = h1_clone->ProjectionY();
    TH1D *h2_proj = h2_clone->ProjectionY();
    TH1D *h3_proj = h3_clone->ProjectionY();
    // TH1D *h4_proj = h4_clone->ProjectionY();

    // Set to appropriate name
    std::string hname;
    hname = "h_rho_jetcone_R" + jetR + "_" + pt_min + "_" + pt_max;
    h1_proj->SetNameTitle(hname.c_str(), hname.c_str());
    hname = "h_rho_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max;
    h2_proj->SetNameTitle(hname.c_str(), hname.c_str());
    hname = "h_rho_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max;
    h3_proj->SetNameTitle(hname.c_str(), hname.c_str());
    hname = "h_rho_wta_jetcone_R" + jetR + "_" + pt_min + "_" + pt_max;
    // h4_proj->SetNameTitle(hname.c_str(), hname.c_str());

    // Rebin
    TH1 *h1 = DivideByBinWidth((TH1 *)h1_proj->Clone(h1_proj->GetName()));
    TH1 *h2 = DivideByBinWidth((TH1 *)h2_proj->Clone(h2_proj->GetName()));
    TH1 *h3 = DivideByBinWidth((TH1 *)h3_proj->Clone(h3_proj->GetName()));
    // TH1 *h4 = DivideByBinWidth((TH1 *)h4_proj->Clone(h4_proj->GetName()));

    // Format color and style
    int markercolor1 = kBlue; // 1
    int markerstyle1 = kFullCircle;
    int markercolor2 = kGreen + 2;
    int markerstyle2 = 33;
    int markercolor3 = kRed;
    int markerstyle3 = 21;
    int markercolor4 = kTeal;
    int markerstyle4 = 24;
    int markercolor5 = kGreen + 3;
    int markerstyle5 = 24;
    int markercolor6 = kRed + 1;
    int markerstyle6 = 24;
    int markercolor7 = kBlack; // inclusive
    int markerstyle7 = 29;

    TH1 *h_JetPt = (TH1 *)f->Get(("h_JetPt_ch_combined_R" + jetR).c_str());//((TH2 *)f->Get(("h_JetPt_ch_combined_vs_pp_R" + jetR).c_str()))->ProjectionY();
    std::cout << "Number of jets from pT bins " << stof(pt_min) / 2 + 1 << "-" << stof(pt_max) / 2 << ": ";
    double njets = h_JetPt->Integral((int)(stof(pt_min) / 2 + 1), (int)(stof(pt_max) / 2));
    std::cout << njets << endl;

    // Format histograms for plotting (this order needed to keep legend in order and graphs lookin good)
    // make a canvas
    TCanvas *c = new TCanvas();
    ProcessCanvas(c);
    c->cd();

    TLegend *l = new TLegend(0.50, 0.48, 0.8562155, 0.8885185, "");

    addLegendInfo(l, ptbin, jetRPoint);
    h1->Scale(1.0 / njets);
    h2->Scale(1.0 / njets);
    h3->Scale(1.0 / njets);
    // h4->Scale(1.0 / njets);
    std::cout << "Mean values: " << h1->GetMean() << ", " << h2->GetMean() << ", " << h3->GetMean() << endl; // << ", " << h4->GetMean() << endl;

    FormatHist(l, h1, "jet cone", markercolor1, markerstyle1);
    // FormatHist(l, h4, "WTA jet cone", markercolor4, markerstyle4);
    FormatHist(l, h2, "perp cone", markercolor2, markerstyle2);
    FormatHist(l, h3, "ME cone", markercolor3, markerstyle3);
    double arr_of_maxes[] = {h1->GetMaximum(), h2->GetMaximum()};
    double &maxy = *std::max_element(arr_of_maxes, arr_of_maxes + 2); // bc there are 4 elements in arr_of_maxes
    std::cout << "the max is " << maxy << endl;
    maxy *= 1.2;

    h1->Rebin(2);
    h2->Rebin(2);
    h3->Rebin(2);
    // h4->Rebin(2);

    h1->GetXaxis()->SetRangeUser(0., 600.);
    h2->GetXaxis()->SetRangeUser(0., 600.);
    h3->GetXaxis()->SetRangeUser(0., 600.);
    // h4->GetXaxis()->SetRangeUser(0., 600.);
    h1->GetYaxis()->SetRangeUser(0, 0.05);
    h1->GetXaxis()->SetTitle("#rho_{local} [GeV]");

    // h1->SetFillStyle(3002);
    // h1->SetFillColor(kBlue);
    h1->Draw("hist");
    h1->Draw("L same");
    h2->Draw("L same");
    h3->Draw("L same");
    // h4->Draw("L same");

    // draw legend
    l->Draw("same");

    std::string fname = outdir + "rho_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".pdf";
    const char *fnamec = fname.c_str();
    c->SaveAs(fnamec);
    delete c;
    delete l;

    // Write rebinned histograms to root file
    f_out->cd();
    h1->Write();
    h2->Write();
    h3->Write();
    // h4->Write();

    f->Close();
    delete f;

    return;
}