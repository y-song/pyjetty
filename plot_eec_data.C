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

void addLegendInfo(TLegend *l, TString ptbin)
{
    l->SetTextSize(0.045);
    l->AddEntry("NULL", "ALICE Pb#minusPb #sqrt{#it{s}_{NN}} = 5.02 TeV, 0#minus10%", "h");
    l->AddEntry("NULL", "charged jets, anti-#it{k}_{T}, #it{R} = 0.4", "h");
    l->AddEntry("NULL", ptbin, "h");
    l->SetTextSize(0.037);
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

void plot_eec_data(string pt_min, string pt_max)
{

    // gROOT->SetBatch(); //prevents plots from showing up
    // std::string JOB_ID = std::to_string(27163555);
    // cout << "JOB_ID IS : " << JOB_ID << endl;

    gStyle->SetOptStat(0);
    SetStyle();
    Double_t markers[10] = {kFullCircle, kFullSquare, kFullDiamond, kFullTriangleUp, kFullStar, kOpenCircle, kOpenTriangleUp, kOpenDiamond, kOpenSquare, kOpenStar};
    Double_t marker_size = 1.5;
    Double_t colors[16] = {kRed, kGreen + 2, kBlue, kBlack, kGreen + 1, kBlue + 1, kRed + 2, kGreen + 2, kBlue + 2, kRed + 3, kGreen + 3, kBlue + 3, kOrange + 1, kViolet + 1, kYellow + 1, kCyan + 1};

    // string one("");
    const char infile[] = "/global/cfs/cdirs/alice/youqi/mypyjetty/pyjetty/AnalysisResults.root";
    // const char infile[] = "/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/youqi/44529118/AnalysisResultsFinal.root";
    TFile *f = new TFile(TString(infile), "READ");
    std::string add_name = "_test";//PbPb_embed_perpcone";
    std::cout << "output name will be " << add_name << std::endl;
    std::string outdir = "";
    std::string outfile = outdir + "AnalysisResultsOut" + add_name + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    const int numjetaxes = 1;
    const string jetR = "04";
    const std::string jetaxis_names[] = {"raw"};
    const std::string hist_names[] = {"jet_ENC_RL2_JetPt_R04_trk10"};
    std::cout << "checkpoint1" << std::endl;

    std::cout << "pt min: " << pt_min << ", pt max: " << pt_max << endl;

    std::vector<std::string> h1_names;
    // std::vector<std::string> h2_names;
    // std::vector<std::string> h3_names;
    // std::vector<std::string> h4_names;
    // std::vector<std::string> h5_names;
    // std::vector<std::string> h6_names;
    // std::vector<std::string> h7_names;

    std::cout << "R = " << jetR << endl;

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        const std::string h1_name = "h_" + hist_names[iobs];
        // const std::string h2_name = "h_" + hist_names[iobs] + "matched_ENC2_sb_JetPt_ch_R" + jetR + "_trk10";
        // const std::string h3_name = "h_" + hist_names[iobs] + "matched_ENC2_bb_JetPt_ch_R" + jetR + "_trk10";
        // const std::string h4_name = "h_" + hist_names[iobs + 3] + "matched_ENC2_ss_JetPt_ch_R" + jetR + "_trk10";
        // const std::string h5_name = "h_" + hist_names[iobs + 3] + "matched_ENC2_sb_JetPt_ch_R" + jetR + "_trk10";
        // const std::string h6_name = "h_" + hist_names[iobs + 3] + "matched_ENC2_bb_JetPt_ch_R" + jetR + "_trk10";
        // const std::string h7_name = "h_" + hist_names[iobs + 6] + "matched_ENC2_sb_JetPt_ch_R" + jetR + "_trk10";

        h1_names.push_back(h1_name);
        // h2_names.push_back(h2_name);
        // h3_names.push_back(h3_name);
        // h4_names.push_back(h4_name);
        // h5_names.push_back(h5_name);
        // h6_names.push_back(h6_name);
        // h7_names.push_back(h7_name);
    }
    std::cout << "checkpoint2" << endl;

    // define pt related variables
    // TString ptbin = pt_min + " < #it{p}_{T, jet}^{det} < " + pt_max + " GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5"; //;TString::Format("%s < #it{p}_{T}^{ch. jet} < %s GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5", pt_min, pt_max);
    TString ptbin = pt_min + " < #it{p}_{T, jet}^{det} < " + pt_max + " GeV/#it{c}, #it{A}_{jet} < 0.6#it{#piR}^{2}"; //;TString::Format("%s < #it{p}_{T}^{ch. jet} < %s GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5", pt_min, pt_max);
    std::string pdf_outdir = "";

    //-------------------------------------------------//
    // find D0 reconstruction through charm
    std::vector<TH2 *> h1_vector;
    // std::vector<TH2 *> h2_vector;
    // std::vector<TH2 *> h3_vector;
    // std::vector<TH2 *> h4_vector;
    // std::vector<TH2 *> h5_vector;
    // std::vector<TH2 *> h6_vector;
    // std::vector<TH2 *> h7_vector;
    
    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        std::cout << h1_names[iobs].c_str() << endl;
        h1_vector.push_back((TH2 *)f->Get(h1_names[iobs].c_str()));
        // h2_vector.push_back((TH2 *)f->Get(h2_names[iobs].c_str()));
        // h3_vector.push_back((TH2 *)f->Get(h3_names[iobs].c_str()));
        // h4_vector.push_back((TH2 *)f->Get(h4_names[iobs].c_str()));
        // h5_vector.push_back((TH2 *)f->Get(h5_names[iobs].c_str()));
        // h6_vector.push_back((TH2 *)f->Get(h6_names[iobs].c_str()));
        // h7_vector.push_back((TH2 *)f->Get(h7_names[iobs].c_str()));
    }
    std::cout << "checkpoint3" << endl;

    std::vector<TH2 *> h1_clones;
    // std::vector<TH2 *> h2_clones;
    // std::vector<TH2 *> h3_clones;
    // std::vector<TH2 *> h4_clones;
    // std::vector<TH2 *> h5_clones;
    // std::vector<TH2 *> h6_clones;
    // std::vector<TH2 *> h7_clones;

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        h1_clones.push_back((TH2 *)h1_vector[iobs]->Clone(Form("%s_clone", h1_names[iobs].c_str())));
        // h2_clones.push_back((TH2 *)h2_vector[iobs]->Clone(Form("%s_clone", h2_names[iobs].c_str())));
        // h3_clones.push_back((TH2 *)h3_vector[iobs]->Clone(Form("%s_clone", h3_names[iobs].c_str())));
        // h4_clones.push_back((TH2 *)h4_vector[iobs]->Clone(Form("%s_clone", h4_names[iobs].c_str())));
        // h5_clones.push_back((TH2 *)h5_vector[iobs]->Clone(Form("%s_clone", h5_names[iobs].c_str())));
        // h6_clones.push_back((TH2 *)h6_vector[iobs]->Clone(Form("%s_clone", h6_names[iobs].c_str())));
        // h7_clones.push_back((TH2 *)h7_vector[iobs]->Clone(Form("%s_clone", h7_names[iobs].c_str())));
    }

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        h1_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
        // h2_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h3_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h4_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h5_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h6_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
        // h7_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
    }

    std::cout << "checkpoint4" << endl;

    // Project onto observable axis
    std::vector<TH1D *> h1_projs;
    // std::vector<TH1D *> h2_projs;
    // std::vector<TH1D *> h3_projs;
    // std::vector<TH1D *> h4_projs;
    // std::vector<TH1D *> h5_projs;
    // std::vector<TH1D *> h6_projs;
    // std::vector<TH1D *> h7_projs;
    // std::vector<TH1D *> htotal_projs;

    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        std::cout << "iobs = " << iobs << endl;
        h1_projs.push_back(h1_clones[iobs]->ProjectionY());
        // h2_projs.push_back(h2_clones[iobs]->ProjectionY());
        // h3_projs.push_back(h3_clones[iobs]->ProjectionY());

        // TH1D *h_total = (TH1D *)h1_clones[iobs]->ProjectionY()->Clone("h_total");
        // h_total->Add(h2_clones[iobs]->ProjectionY());
        // h_total->Add(h3_clones[iobs]->ProjectionY());
        // htotal_projs.push_back(h_total);
        
        // TH1D *h5 = h5_clones[iobs]->ProjectionY();
        // TH1D *h6 = h6_clones[iobs]->ProjectionY();
        // TH1D *h7 = h7_clones[iobs]->ProjectionY();

        // double rho_ratio = 75.95/70.01;
        
        // h5->Scale(rho_ratio);
        // h6->Scale(rho_ratio*rho_ratio);
        // h7->Scale(rho_ratio*rho_ratio);
        
        //h5->Add(h7, -1); // 2 perp cones
        // h5->Add(h3_clones[iobs]->ProjectionY(), -2); // 1 perp cone
        // h5_projs.push_back(h5); // sig-bkg
        
        // h6_projs.push_back(h6); // bkg-bkg

        // TH1D *h4 = (TH1D *)h_total->Clone("h4");
        // h4->Add(h5, -1);
        // h4->Add(h6, -1);
        // h4_projs.push_back(h4); // all - bb - (2)sb
    }

    // Set to appropriate name
    std::string hname;
    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        hname = h1_projs[iobs]->GetName();
        hname += "_pt" + pt_min + "-" + pt_max;
        h1_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = h2_projs[iobs]->GetName();
        // hname += "_pt" + pt_min + "-" + pt_max;
        // h2_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = h3_projs[iobs]->GetName();
        // hname += "_pt" + pt_min + "-" + pt_max;
        // h3_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = h4_projs[iobs]->GetName();
        // hname += "_pt" + pt_min + "-" + pt_max;
        // h4_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = h5_projs[iobs]->GetName();
        // hname += "_pt" + pt_min + "-" + pt_max;
        // h5_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
        // hname = h6_projs[iobs]->GetName();
        // hname += "_pt" + pt_min + "-" + pt_max;
        // h6_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
    }

    // Rebin
    std::vector<TH1 *> h1;
    // std::vector<TH1 *> h2;
    // std::vector<TH1 *> h3;
    // std::vector<TH1 *> h4;
    // std::vector<TH1 *> h5;
    // std::vector<TH1 *> h6;
    // std::vector<TH1 *> htotal;

    std::cout << "checkpoint6" << endl;
    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        std::string h1_newname = h1_projs[iobs]->GetName();
        // std::string h2_newname = h2_projs[iobs]->GetName();
        // std::string h3_newname = h3_projs[iobs]->GetName();
        // std::string h4_newname = h4_projs[iobs]->GetName();
        // std::string h5_newname = h5_projs[iobs]->GetName();
        // std::string h6_newname = h6_projs[iobs]->GetName();
        // std::string htotal_newname = htotal_projs[iobs]->GetName();

        h1.push_back(DivideByBinWidth((TH1 *)h1_projs[iobs]->Clone(h1_newname.c_str()))); // h1_newname + "rebin").c_str() );
        // h2.push_back(DivideByBinWidth((TH1 *)h2_projs[iobs]->Clone(h2_newname.c_str()))); // h2_newname + "rebin").c_str() );
        // h3.push_back(DivideByBinWidth((TH1 *)h3_projs[iobs]->Clone(h3_newname.c_str())));
        // h4.push_back(DivideByBinWidth((TH1 *)h4_projs[iobs]->Clone(h4_newname.c_str())));
        // h5.push_back(DivideByBinWidth((TH1 *)h5_projs[iobs]->Clone(h5_newname.c_str())));
        // h6.push_back(DivideByBinWidth((TH1 *)h6_projs[iobs]->Clone(h6_newname.c_str())));
        // htotal.push_back(DivideByBinWidth((TH1 *)htotal_projs[iobs]->Clone(htotal_newname.c_str())));
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

    TH1 *h_JetPt = ((TH2 *)f->Get("h_Nconst_JetPt_R04_trk10"))->ProjectionY();
    std::cout << "Number of jets from pT bins " << stof(pt_min) + 1 << "-" << stof(pt_max)<< ": ";
    int njets = h_JetPt->Integral((int)(stof(pt_min) + 1), (int)(stof(pt_max)));
    std::cout << njets << endl;

    // Format histograms for plotting (this order needed to keep legend in order and graphs lookin good)
    for (int iobs = 0; iobs < numjetaxes; iobs++)
    {
        // make a canvas
        TCanvas *c = new TCanvas();
        ProcessCanvas(c);
        c->cd();
        gPad->SetLogx();

        TLegend *l = new TLegend(0.17, 0.4, 0.5, 0.88);
        TLegend *l2 = new TLegend(0, 0.7, 0.55, 0.9);
        l2->SetTextSize(0.037);
        l2->SetBorderSize(0);

        addLegendInfo(l, ptbin);
        h1[iobs]->Scale(1.0 / njets);
        // h2[iobs]->Scale(1.0 / njets);
        // h3[iobs]->Scale(1.0 / njets);
        // h4[iobs]->Scale(1.0 / njets);
        // h5[iobs]->Scale(1.0 / njets);
        // h6[iobs]->Scale(1.0 / njets);
        // htotal[iobs]->Scale(1.0 / njets);

        FormatHist(l, h1[iobs], "raw", markercolor1, markerstyle1);
        // FormatHist(l, h2[iobs], "sig-bkg", markercolor2, markerstyle2);
        // FormatHist(l, h3[iobs], "bkg-bkg", markercolor3, markerstyle3);
        // FormatHist(l, htotal[iobs], "all comb.", markercolor7, markerstyle7);
        // FormatHist(l2, h4[iobs], "#splitline{all comb.}{#minus [0.5(jet-perp #minus 2perp-perp) + perp-perp]}", markercolor4, markerstyle4);
        // FormatHist(l2, h5[iobs], "0.5(jet-perp #minus 2perp-perp)", markercolor5, markerstyle5);
        // FormatHist(l2, h6[iobs], "perp-perp", markercolor6, markerstyle6);
        // double arr_of_maxes[] = {h1[iobs]->GetMaximum(), h2[iobs]->GetMaximum()};
        // double &maxy = *std::max_element(arr_of_maxes, arr_of_maxes + 2); // bc there are 4 elements in arr_of_maxes
        // std::cout << "the max is " << maxy << endl;
        // maxy *= 1.2;
        // h1[iobs]->SetMaximum(maxy);
        h1[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h2[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h3[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h4[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h5[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // h6[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        // htotal[iobs]->GetXaxis()->SetRangeUser(0.005, 0.4);
        h1[iobs]->GetYaxis()->SetRangeUser(0, 10);
        h1[iobs]->GetXaxis()->SetTitle("#it{R}_{L}");
        h1[iobs]->GetYaxis()->SetTitle("EEC");

        h1[iobs]->Draw("L");
        // h2[iobs]->Draw("L same");
        // h3[iobs]->Draw("L same");
        // h4[iobs]->Draw("L same");
        // h5[iobs]->Draw("L same");
        // h6[iobs]->Draw("L same");
        // htotal[iobs]->Draw("L same");

        // draw legend
        l->Draw("same");
        // l2->Draw("same");

        std::string fname = pdf_outdir + jetaxis_names[iobs] + "_pt" + pt_min + '-' + pt_max + "_R" + jetR + add_name + ".pdf";
        const char *fnamec = fname.c_str();
        c->SaveAs(fnamec);
        delete c;
        delete l;
        delete l2;

        // Write rebinned histograms to root file
        f_out->cd();
        h1[iobs]->Write();
        // h2[iobs]->Write();
        // h3[iobs]->Write();
        // h4[iobs]->Write();
        // h5[iobs]->Write();
        // h6[iobs]->Write();
        // htotal[iobs]->Write();
    } // obs bins loop

    f->Close();
    delete f;

    return;
}