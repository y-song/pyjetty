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

void addLegendInfo(TLegend *l, TString ptbin, TString jetR)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "PYTHIA8 jets + PbPb 0#minus10%", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV", "h");
    l->AddEntry("NULL", "charged jets, anti-#it{k}_{T}, #it{R} = " + jetR, "h");
    l->AddEntry("NULL", ptbin, "h");
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

void plot_eec_mbcone(string pt_min, string pt_max)
{

    gStyle->SetOptStat(0);
    SetStyle();

    const char infile[] = "/global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/AnalysisResults.root";
    // const char infile[] = "/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/48199478/AnalysisResultsFinal.root";
    const string jetR = "02";
    const string jetRPoint = "0.2";
    std::string add_name = "_test";
    std::string outfile = "eec_mbcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".root";
   
    TFile *f = new TFile(TString(infile), "READ");
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << endl;
    std::cout << "R = " << jetR << endl;
    std::cout << "checkpoint1" << std::endl;

    const int numjetaxes = 2;
    const std::string jetaxis_names[] = {"jet", "matched_jet"};
    const std::string hist_names[] = {"", "matched_", "mbcone"+jetRPoint+"_", "matched_mbcone"+jetRPoint+"_", "2mbcone"+jetRPoint+"_", "matched_2mbcone"+jetRPoint+"_"};

    std::vector<TH1D *> h1;
    std::vector<TH1D *> h2;
    std::vector<TH1D *> h3;
    std::vector<TH1D *> h4;
    std::vector<TH1D *> h5;
    std::vector<TH1D *> h6;
    std::vector<TH1D *> h7;
    std::vector<TH1D *> htotal;

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        const std::string h1_name = "h_" + hist_names[iobs] + "ENC2_ss_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h2_name = "h_" + hist_names[iobs] + "ENC2_sb_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h3_name = "h_" + hist_names[iobs] + "ENC2_bb_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h4_name = "h_" + hist_names[iobs + 2] + "ENC2_ss_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h5_name = "h_" + hist_names[iobs + 2] + "ENC2_sb_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h6_name = "h_" + hist_names[iobs + 2] + "ENC2_bb_JetPt_ch_combined_R" + jetR + "_trk10";
        const std::string h7_name = "h_" + hist_names[iobs + 4] + "ENC2_sb_JetPt_ch_combined_R" + jetR + "_trk10";
        
        std::cout << "iobs: " << iobs << endl;
        std::cout << "h1: " << h1_name.c_str() << endl;
        std::cout << "h2: " << h2_name.c_str() << endl;
        std::cout << "h3: " << h3_name.c_str() << endl;
        std::cout << "h4: " << h4_name.c_str() << endl;
        std::cout << "h5: " << h5_name.c_str() << endl;
        std::cout << "h6: " << h6_name.c_str() << endl;

        // clone and process histograms
        TH2 *h1_clone = (TH2 *)f->Get(h1_name.c_str())->Clone(Form("%s_clone", h1_name.c_str()));
        TH2 *h2_clone = (TH2 *)f->Get(h2_name.c_str())->Clone(Form("%s_clone", h2_name.c_str()));
        TH2 *h3_clone = (TH2 *)f->Get(h3_name.c_str())->Clone(Form("%s_clone", h3_name.c_str()));
        TH2 *h4_clone = (TH2 *)f->Get(h4_name.c_str())->Clone(Form("%s_clone", h4_name.c_str()));
        TH2 *h5_clone = (TH2 *)f->Get(h5_name.c_str())->Clone(Form("%s_clone", h5_name.c_str()));
        TH2 *h6_clone = (TH2 *)f->Get(h6_name.c_str())->Clone(Form("%s_clone", h6_name.c_str()));
        TH2 *h7_clone = (TH2 *)f->Get(h7_name.c_str())->Clone(Form("%s_clone", h7_name.c_str()));
        h1_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
        h2_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        h3_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        h4_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        h5_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        h6_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        h7_clone->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        TH1D *h1_proj = h1_clone->ProjectionY();
        TH1D *h2_proj = h2_clone->ProjectionY();
        TH1D *h3_proj = h3_clone->ProjectionY();
        TH1D *h4_proj = h4_clone->ProjectionY();
        TH1D *h5_proj = h5_clone->ProjectionY();
        TH1D *h6_proj = h6_clone->ProjectionY();
        TH1D *h7_proj = h7_clone->ProjectionY();

        TH1D *h_total_proj = (TH1D *)h1_proj->Clone("h_total_proj");
        h_total_proj->Add(h2_proj);
        h_total_proj->Add(h3_proj);

        // double rho_ratio = 1.0; // 75.95/70.01;
        // h5_proj->Scale(rho_ratio);
        // h6_proj->Scale(rho_ratio*rho_ratio); // bkg-bkg
        // h7_proj->Scale(rho_ratio*rho_ratio);

        h5_proj->Add(h7_proj, -1); // 2 mb cones, sig-bkg
        // h5->Add(h6, -2); // 1 mb cone
                
        h4_proj->Add(h5_proj, -1);
        h4_proj->Add(h6_proj, -1); // all - bb - (2)sb
    
        h1.push_back(DivideByBinWidth((TH1D *)h1_proj->Clone(h1_proj->GetName())));
        h2.push_back(DivideByBinWidth((TH1D *)h2_proj->Clone(h2_proj->GetName())));
        h3.push_back(DivideByBinWidth((TH1D *)h3_proj->Clone(h3_proj->GetName())));
        h4.push_back(DivideByBinWidth((TH1D *)h4_proj->Clone(h4_proj->GetName())));
        h5.push_back(DivideByBinWidth((TH1D *)h5_proj->Clone(h5_proj->GetName())));
        h6.push_back(DivideByBinWidth((TH1D *)h6_proj->Clone(h6_proj->GetName())));
        htotal.push_back(DivideByBinWidth((TH1D *)h_total_proj->Clone(h_total_proj->GetName())));
    }
    
    // Set to appropriate name
    std::string hname;
    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        hname = "h_eec_ss_" + hist_names[iobs] + "truth_R" + jetR + "_" + pt_min + "_" + pt_max;
        h1[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_eec_sb_" + hist_names[iobs] + "truth_R" + jetR + "_" + pt_min + "_" + pt_max;
        h2[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_eec_bb_" + hist_names[iobs] + "truth_R" + jetR + "_" + pt_min + "_" + pt_max;
        h3[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_eec_ss_" + hist_names[iobs] + "mbcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        h4[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_eec_sb_" + hist_names[iobs] + "mbcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        h5[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        hname = "h_eec_bb_" + hist_names[iobs] + "mbcone_R" + jetR + "_" + pt_min + "_" + pt_max;
        h6[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
    }

    // Format color and style
    int markercolor1 = kBlue; // 1
    int markerstyle1 = kFullCircle;
    int markercolor2 = kGreen + 2;
    int markerstyle2 = 33;
    int markercolor3 = kRed;
    int markerstyle3 = 21;
    int markercolor4 = kBlue + 1;
    int markerstyle4 = 24;
    int markercolor5 = kGreen + 3;
    int markerstyle5 = 24;
    int markercolor6 = kRed + 1;
    int markerstyle6 = 24;
    int markercolor7 = kBlack; // inclusive
    int markerstyle7 = 29;

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        // normalize
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
        h1[iobs]->Scale(1.0 / njets);
        h2[iobs]->Scale(1.0 / njets);
        h3[iobs]->Scale(1.0 / njets);
        h4[iobs]->Scale(1.0 / njets);
        h5[iobs]->Scale(1.0 / njets);
        h6[iobs]->Scale(1.0 / njets);
        htotal[iobs]->Scale(1.0 / njets);

        // make a canvas
        TCanvas *c = new TCanvas();
        ProcessCanvas(c);
        c->cd();
        gPad->SetLogx();

        TLegend *l = new TLegend(0.17, 0.4, 0.5, 0.88);
        TLegend *l2 = new TLegend(0, 0.7, 0.55, 0.9);
        l2->SetTextSize(0.037);
        l2->SetBorderSize(0);

        TString ptbin = pt_min + " < #it{p}_{T}^{combined jet, sub.} < " + pt_max + " GeV/#it{c}, #it{A}_{jet} > 0.6#it{#piR}^{2}"; //;TString::Format("%s < #it{p}_{T}^{ch. jet} < %s GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5", pt_min, pt_max);
        addLegendInfo(l, ptbin, jetRPoint);


        FormatHist(l, h1[iobs], "sig-sig", markercolor1, markerstyle1);
        FormatHist(l, h2[iobs], "sig-bkg", markercolor2, markerstyle2);
        FormatHist(l, h3[iobs], "bkg-bkg", markercolor3, markerstyle3);
        FormatHist(l, htotal[iobs], "all comb.", markercolor7, markerstyle7);
        FormatHist(l2, h4[iobs], "#splitline{all comb.}{#minus [0.5(jet-perp #minus 2perp-perp) + perp-perp]}", markercolor4, markerstyle4);
        FormatHist(l2, h5[iobs], "0.5(jet-perp #minus 2perp-perp)", markercolor5, markerstyle5);
        FormatHist(l2, h6[iobs], "perp-perp", markercolor6, markerstyle6);
        double arr_of_maxes[] = {h1[iobs]->GetMaximum(), h2[iobs]->GetMaximum()};
        double &maxy = *std::max_element(arr_of_maxes, arr_of_maxes + 2); // bc there are 4 elements in arr_of_maxes
        std::cout << "the max is " << maxy << endl;
        maxy *= 1.2;
        h1[iobs]->SetMaximum(maxy);
        h1[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h2[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h3[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h4[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h5[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h6[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        htotal[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h1[iobs]->GetYaxis()->SetRangeUser(0, 10);
        h1[iobs]->GetXaxis()->SetTitle("#it{R}_{L}");

        h1[iobs]->Draw("L");
        h2[iobs]->Draw("L same");
        h3[iobs]->Draw("L same");
        h4[iobs]->Draw("L same");
        h5[iobs]->Draw("L same");
        h6[iobs]->Draw("L same");
        htotal[iobs]->Draw("L same");

        // draw legend
        l->Draw("same");
        // l2->Draw("same");

        // save canvas to pdf
        std::string fname = jetaxis_names[iobs] + "_eec_mbcone_embed_PbPb_R" + jetR + "_" + pt_min + "_" + pt_max + add_name + ".pdf";
        const char *fnamec = fname.c_str();
        c->SaveAs(fnamec);
        delete c;
        delete l;
        delete l2;

        // save histograms to root file
        f_out->cd();
        h1[iobs]->Write();
        h2[iobs]->Write();
        h3[iobs]->Write();
        h4[iobs]->Write();
        h5[iobs]->Write();
        h6[iobs]->Write();
        htotal[iobs]->Write();
    } // obs bins loop

    f->Close();
    delete f;

    return;
}