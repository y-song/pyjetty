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
    l->AddEntry("NULL", "PYTHIA8 leading jets + thermal", "h");
    l->AddEntry("NULL", "#sqrt{#it{s}} = 5.02 TeV, #hat{#it{p}}_{T} > 30 GeV", "h");
    l->AddEntry("NULL", "charged jets, anti-#it{k}_{T}, #it{R} = 4", "h");
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

        output_hist->SetBinContent(ibin, bincontent/binwidth);
        output_hist->SetBinError(ibin, binerror/binwidth);
    }
    return output_hist;
}

void plot_eec(string pt_min, string pt_max)
{

    // gROOT->SetBatch(); //prevents plots from showing up
    // std::string JOB_ID = std::to_string(27163555);
    // cout << "JOB_ID IS : " << JOB_ID << endl;

    gStyle->SetOptStat(0);
    SetStyle();
    Double_t markers[10] = {kFullCircle, kFullSquare, kFullDiamond, kFullTriangleUp, kFullStar, kOpenCircle, kOpenTriangleUp, kOpenDiamond, kOpenSquare, kOpenStar};
    Double_t marker_size = 1.5;
    Double_t colors[16] = {kRed, kGreen + 2, kBlue, kRed + 1, kGreen + 1, kBlue + 1, kRed + 2, kGreen + 2, kBlue + 2, kRed + 3, kGreen + 3, kBlue + 3, kOrange + 1, kViolet + 1, kYellow + 1, kCyan + 1};

    // string one("");
    const char infile[] = "/global/cfs/cdirs/alice/youqi/AnalysisResultsR04_eec_jets_in_thermal.root";
    TFile *f = new TFile(TString(infile), "READ");
    std::string add_name = "_test";
    std::cout << "output name will be " << add_name << std::endl;
    std::string outdir = "";
    std::string outfile = outdir + "AnalysisResultsOut" + add_name + ".root";
    TFile *f_out = new TFile(outfile.c_str(), "RECREATE");

    const int numjetaxes = 3;
    const int n_bins = 2;
    std::string jetR_list[] = {"04"};
    const std::string jetaxis_names[] = {"jet", "jetcone", "wta_jetcone"};
    const std::string hist_names[] = {"", "jetcone0.4_", "wta_jetcone0.4_"};
    std::cout << "checkpoint1" << std::endl;

    for (int i = 0; i < n_bins; i++)
    {
        cout << "pt min: " << pt_min << ", pt max: " << pt_max << endl;

        for (std::string jetR : jetR_list)
        {

            std::vector<std::string> h1_names;
            std::vector<std::string> h2_names;
            std::vector<std::string> h3_names;
            cout << "R = " << jetR << endl;

            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                const std::string h1_name = "h_" + hist_names[iobs] + "matched_ENC2_ss_JetPt_ch_R" + jetR + "_trk10";
                const std::string h2_name = "h_" + hist_names[iobs] + "matched_ENC2_sb_JetPt_ch_R" + jetR + "_trk10";
                const std::string h3_name = "h_" + hist_names[iobs] + "matched_ENC2_bb_JetPt_ch_R" + jetR + "_trk10";

                h1_names.push_back(h1_name);
                h2_names.push_back(h2_name);
                h3_names.push_back(h3_name);
            }
            cout << "checkpoint2" << endl;

            // define pt related variables
            TString ptbin = pt_min + " < #it{p}_{T}^{ch. jet} < " + pt_max + " GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5"; //;TString::Format("%s < #it{p}_{T}^{ch. jet} < %s GeV/#it{c}, #font[122]{|}#it{#eta}_{jet}#font[122]{|} #leq 0.5", pt_min, pt_max);
            std::string pdf_outdir = "";

            //-------------------------------------------------//
            // find D0 reconstruction through charm
            std::vector<TH2 *> h1_vector;
            std::vector<TH2 *> h2_vector;
            std::vector<TH2 *> h3_vector;

            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                cout << h1_names[iobs].c_str() << endl;
                h1_vector.push_back((TH2 *)f->Get(h1_names[iobs].c_str()));
                h2_vector.push_back((TH2 *)f->Get(h2_names[iobs].c_str()));
                h3_vector.push_back((TH2 *)f->Get(h3_names[iobs].c_str()));
            }
            cout << "checkpoint3" << endl;

            std::vector<TH2 *> h1_clones;
            std::vector<TH2 *> h2_clones;
            std::vector<TH2 *> h3_clones;

            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                h1_clones.push_back((TH2 *)h1_vector[iobs]->Clone(Form("%s_clone", h1_names[iobs].c_str())));
                h2_clones.push_back((TH2 *)h2_vector[iobs]->Clone(Form("%s_clone", h2_names[iobs].c_str())));
                h3_clones.push_back((TH2 *)h3_vector[iobs]->Clone(Form("%s_clone", h3_names[iobs].c_str())));
            }

            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                h1_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max)); // apply cut on jet pt
                h2_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
                h3_clones[iobs]->GetXaxis()->SetRangeUser(stof(pt_min), stof(pt_max));
            }

            cout << "checkpoint4" << endl;

            // Project onto observable axis
            std::vector<TH1D *> h1_projs;
            std::vector<TH1D *> h2_projs;
            std::vector<TH1D *> h3_projs;

            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                h1_projs.push_back(h1_clones[iobs]->ProjectionY());
                h2_projs.push_back(h2_clones[iobs]->ProjectionY());
                h3_projs.push_back(h3_clones[iobs]->ProjectionY());
            }

            // Set to appropriate name
            std::string hname;
            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                hname = h1_projs[iobs]->GetName();
                hname += "_pt" + pt_min + "-" + pt_max;
                h1_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
                hname = h2_projs[iobs]->GetName();
                hname += "_pt" + pt_min + "-" + pt_max;
                h2_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
                hname = h3_projs[iobs]->GetName();
                hname += "_pt" + pt_min + "-" + pt_max;
                h3_projs[iobs]->SetNameTitle(hname.c_str(), hname.c_str());
            }

            // Rebin
            std::vector<TH1 *> h1;
            std::vector<TH1 *> h2;
            std::vector<TH1 *> h3;

            cout << "checkpoint6" << endl;
            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                std::string h1_newname = h1_projs[iobs]->GetName();
                std::string h2_newname = h2_projs[iobs]->GetName();
                std::string h3_newname = h3_projs[iobs]->GetName();

                h1.push_back(DivideByBinWidth( (TH1 *)h1_projs[iobs]->Clone(h1_newname.c_str()) )); // h1_newname + "rebin").c_str() );
                h2.push_back(DivideByBinWidth( (TH1 *)h2_projs[iobs]->Clone(h2_newname.c_str()) )); // h2_newname + "rebin").c_str() );
                h3.push_back(DivideByBinWidth( (TH1 *)h3_projs[iobs]->Clone(h3_newname.c_str()) ));
            }

            // Format color and style
            int markercolor1 = kBlue; // 1
            int markerstyle1 = kFullCircle;
            int markercolor2 = kGreen + 2;
            int markerstyle2 = 33;
            int markercolor3 = kRed;
            int markerstyle3 = 21;
            int markercolor4 = kMagenta - 7; // inclusive
            int markerstyle4 = 29;
            int markercolor5 = kRed; // inclusive
            int markerstyle5 = 24;

            // Format histograms for plotting (this order needed to keep legend in order and graphs lookin good)
            for (int iobs = 0; iobs < numjetaxes; iobs++)
            {
                // make a canvas
                TCanvas *c = new TCanvas();
                ProcessCanvas(c);
                c->cd();
                gPad->SetLogx();

                TLegend *l = new TLegend(0.17, 0.4, 0.5, 0.88);

                addLegendInfo(l, ptbin);
                FormatHist(l, h1[iobs], "sig-sig", markercolor1, markerstyle1);
                FormatHist(l, h2[iobs], "sig-bkg", markercolor2, markerstyle2);
                FormatHist(l, h3[iobs], "bkg-bkg", markercolor3, markerstyle3);
                double arr_of_maxes[] = {h1[iobs]->GetMaximum(), h2[iobs]->GetMaximum()};
                double &maxy = *std::max_element(arr_of_maxes, arr_of_maxes + 2); // bc there are 4 elements in arr_of_maxes
                cout << "the max is " << maxy << endl;
                maxy *= 1.2;
                h1[iobs]->SetMaximum(maxy);
                h1[iobs]->GetXaxis()->SetRangeUser(0.005, 1);
                h2[iobs]->GetXaxis()->SetRangeUser(0.005, 1);
                h3[iobs]->GetXaxis()->SetRangeUser(0.005, 1);
                h1[iobs]->GetYaxis()->SetRangeUser(0, 80000);
                h1[iobs]->GetXaxis()->SetTitle("#it{R}_{L}");

                h1[iobs]->Draw("L");
                h2[iobs]->Draw("L same");
                h3[iobs]->Draw("L same");

                // draw legend
                l->Draw("same");

                std::string fname = pdf_outdir + jetaxis_names[iobs] + "_pt" + pt_min + '-' + pt_max + "_R" + jetR + add_name + ".pdf";
                const char *fnamec = fname.c_str();
                c->SaveAs(fnamec);
                delete c;
                delete l;

                // Write rebinned histograms to root file
                f_out->cd();
                h1[iobs]->Write();
                h2[iobs]->Write();
                h3[iobs]->Write();
            } // obs bins loop
        } // jet R loop
    } // jet pT loop

    f->Close();
    delete f;

    return;
}