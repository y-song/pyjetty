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
    
    gStyle->SetOptStat(0);
    SetStyle();
    
    // const char infile[] = "/global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/AnalysisResults.root";
    const char infile[] = "/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/48695459/AnalysisResultsFinal.root";
    const string jetR = "02";
    const string jetRPoint = "0.2";
    std::string add_name = "_48695459";
    std::string outfile = "rho_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".root";

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << endl;
    std::cout << "R = " << jetR << endl;  

    TFile *f = new TFile(TString(infile), "READ");
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    const int numjetaxes = 2;
    const std::string jetaxis_names[] = {"jet", "matched_jet"};
    const std::string hist_names[] = {"", "matched_", "mbcone"+jetRPoint+"_", "matched_mbcone"+jetRPoint+"_"};//, "perpcone"+jetRPoint+"_", "matched_perpcone"+jetRPoint+"_"};

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        const std::string h1_name = "h_" + hist_names[iobs] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h2_name = "h_" + hist_names[iobs+2] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";
        // const std::string h3_name = "h_" + hist_names[iobs+4] + "rho_local_JetPt_ch_combined_R" + jetR + "_trk10";

        std::cout << "iobs: " << iobs << endl;
        std::cout << "h1: " << h1_name.c_str() << endl;
        std::cout << "h2: " << h2_name.c_str() << endl;
        // std::cout << "h3: " << h2_name.c_str() << endl;

        // clone and process histograms
        TH2 *h1_clone = (TH2 *)f->Get(h1_name.c_str())->Clone(Form("%s_clone", h1_name.c_str()));
        TH2 *h2_clone = (TH2 *)f->Get(h2_name.c_str())->Clone(Form("%s_clone", h2_name.c_str()));
        // TH2 *h3_clone = (TH2 *)f->Get(h3_name.c_str())->Clone(Form("%s_clone", h3_name.c_str()));
        h1_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
        h2_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h3_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        TH1D *h1_proj = h1_clone->ProjectionY();
        TH1D *h2_proj = h2_clone->ProjectionY();
        // TH1D *h3_proj = h3_clone->ProjectionY();
        std::string hname;
        hname = "h_rho_jetcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        h1_proj->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_rho_mbcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        h2_proj->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = "h_rho_perpcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        // h3_proj->SetNameTitle(hname.c_str(), hname.c_str());
        TH1 *h1 = DivideByBinWidth((TH1 *)h1_proj->Clone(h1_proj->GetName()));
        TH1 *h2 = DivideByBinWidth((TH1 *)h2_proj->Clone(h2_proj->GetName()));
        // TH1 *h3 = DivideByBinWidth((TH1 *)h3_proj->Clone(h3_proj->GetName()));
        std::cout << "Integrals: " << h1->Integral() << ", " << h2->Integral() << endl; // ", " << h3->Integral() << endl; // ", " << h4->GetMean() << endl;
        std::cout << "Mean values: " << h1->GetMean() << ", " << h2->GetMean() << endl; // ", " << h3->GetMean() << endl; // ", " << h4->GetMean() << endl;
   
        // get normalization
        TH1 *h_JetPt;
        if (iobs == 0)
        {
            h_JetPt = (TH1 *)f->Get(("h_JetPt_ch_combined_R" + jetR).c_str());
        }
        else if (iobs == 1)
        {
            h_JetPt = (TH1 *)f->Get(("h_matched_JetPt_ch_combined_R" + jetR).c_str());            
        }
        std::cout << "Number of jets from pT bins " << stof(pt_min) / 2 + 1 << "-" << stof(pt_max) / 2 << ": ";
        double njets = h_JetPt->Integral((int)(stof(pt_min) / 2 + 1), (int)(stof(pt_max) / 2));
        std::cout << njets << endl;
        h1->Scale(1.0 / njets);
        h2->Scale(1.0 / njets);
        // h3->Scale(1.0 / njets);

        // make a canvas
        TCanvas *c = new TCanvas();
        ProcessCanvas(c);
        c->cd();

        TLegend *l = new TLegend(0.50, 0.48, 0.8562155, 0.8885185, "");
        TString ptbin = pt_min + " < #it{p}_{T}^{combined jet, sub.} < " + pt_max + " GeV";
        addLegendInfo(l, ptbin, jetRPoint);

        FormatHist(l, h1, TString::Format("jet cone (%.1f)", h1->GetMean()), markercolor1, markerstyle1);
        FormatHist(l, h2, TString::Format("ME cone (%.1f)", h2->GetMean()), markercolor2, markerstyle2);
        // FormatHist(l, h3, TString::Format("perp cone (%.1f)", h3->GetMean()), markercolor3, markerstyle3);

        h1->Rebin(2);
        h2->Rebin(2);
        // h3->Rebin(2);
        h1->GetXaxis()->SetRangeUser(0., 600.);
        h2->GetXaxis()->SetRangeUser(0., 600.);
        // h3->GetXaxis()->SetRangeUser(0., 600.);
        h1->GetYaxis()->SetRangeUser(0, 0.035);
        h1->GetXaxis()->SetTitle("#rho_{local} [GeV]");

        h1->Draw("hist");
        h1->Draw("L same");
        h2->Draw("L same");
        // h3->Draw("L same");
        l->Draw("same");

        // save canvas to pdf
        std::string fname = jetaxis_names[iobs] + "_rho_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".pdf";
        const char *fnamec = fname.c_str();
        c->SaveAs(fnamec);
        delete c;
        delete l;

        // save histograms to root file
        f_out->cd();
        h1->Write();
        h2->Write();
        // h3->Write();
    }

    f->Close();
    delete f;

    return;
}