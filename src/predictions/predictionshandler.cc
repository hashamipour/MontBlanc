//
// Authors: Rabah Abdul Khalek: rabah.khalek@gmail.com
//          Valerio Bertone: valerio.bertone@cern.ch
//          Emanuele R. Nocera: emanuele.nocera@ed.ac.uk
//

#include "MontBlanc/predictionshandler.h"

#include <LHAPDF/LHAPDF.h>
#include <apfel/SIDIS.h>
#include <apfel/zeromasscoefficientfunctionsunp_tl.h>
#include <numeric>

namespace MontBlanc
{
  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(YAML::Node                                     const& config,
                                         NangaParbat::DataHandler                       const& DH,
                                         std::shared_ptr<const apfel::Grid>             const& g,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(config["mu0"].as<double>()),
    _Thresholds(config["thresholds"].as<std::vector<double>>()),
    _g(g),
    _obs(DH.GetObservable()),
    _bins(DH.GetBinning()),
    _qTfact(DH.GetKinematics().qTfact),
    _cmap(apfel::DiagonalBasis{13})
  {
    // Set silent mode for both apfel=+ and LHAPDF;
    apfel::SetVerbosityLevel(0);
    LHAPDF::setVerbosity(0);

    // Charge map used to discriminate between positive, negative and
    // sum of charged hadrons.
    const int charge = DH.GetCharge();
    if (charge == 1)
      // For positive hadrons, being them the default, the charge map
      // is all one's
      _ChargeMap.resize(13, 1);
    else if (charge == -1)
      {
        // For negative hadrons, charge conjugation leaves the sea-like
        // distributions (Sigma, T3, T8, ...) unchanged but changes the
        // sign of the valence-like ones (V, V3, V8, etc.).
        _ChargeMap.resize(13, 1);
        for (int i = 1; i <= 6; i++)
          _ChargeMap[2 * i] = - 1;
      }
    else if (charge == 0)
      {
        // For the sum of positive and negative hadrons, total-like
        // distributions (Sigma, T3, T8, ...) double while valence-like
        // distributions (V, V3, V8, etc.) cancel.
        _ChargeMap.resize(13, 2);
        for (int i = 1; i <= 6; i++)
          _ChargeMap[2 * i] = 0;
      }
    else
      throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unsupported charge.");

    // Perturbative order
    const int PerturbativeOrder = config["perturbative order"].as<int>();

    // Alpha_s
    const apfel::TabulateObject<double> TabAlphas{*(new apfel::AlphaQCD
      {config["alphas"]["aref"].as<double>(), config["alphas"]["Qref"].as<double>(), _Thresholds, PerturbativeOrder}), 100, 0.9, 1001, 3};
    const auto Alphas = [&] (double const& mu) -> double{ return TabAlphas.Evaluate(mu); };

    // Electromagnetic coupling
    const apfel::AlphaQED alphaem{config["alphaem"]["aref"].as<double>(), config["alphaem"]["Qref"].as<double>(), _Thresholds, {0, 0, 1.777}, 0};
    const auto Alphaem = [&] (double const& mu) -> double{ return alphaem.Evaluate(mu); };

    // Initialize QCD time-like evolution operators and tabulated them
    const std::unique_ptr<const apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabGammaij{new const apfel::TabulateObject<apfel::Set<apfel::Operator>>
      {*(BuildDglap(InitializeDglapObjectsQCDT(*_g, _Thresholds, true), _mu0, PerturbativeOrder, Alphas)), 100, 1, 100, 3}};

    // Zero operator
    const apfel::Operator Zero{*_g, apfel::Null{}};

    // Set cuts in the mother class
    this->_cuts = cuts;

    // Compute total cut mask as a product of single masks
    _cutmask.resize(_bins.size(), true);
    for (auto const& c : _cuts)
      _cutmask *= c->GetMask();

    // Center of mass energy
    const double Vs = DH.GetKinematics().Vs;

    // Overall prefactor
    const double pref = DH.GetPrefactor();

    if (DH.GetProcess() == NangaParbat::DataHandler::Process::SIA)
      {
        // Get the strong coupling
        const double as = Alphas(Vs);

        // Get fine-structure constant
        const double aem = Alphaem(Vs);

        // Get evolution-operator objects
        std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Vs).GetObjects();

        // Get F2 objects at the scale Vs
        const apfel::StructureFunctionObjects F2Obj = apfel::InitializeF2NCObjectsZMT(*_g, _Thresholds)(Vs, apfel::ElectroWeakCharges(Vs, true));

        // Get skip vector
        const std::vector<int> skip = F2Obj.skip;

        // Intialise container for the FK table
        std::map<int, apfel::Operator> Cj;
        for (int j = 0; j < 13; j++)
          Cj.insert({j, Zero});

        // Initialise total cross section
        double xsec = 0;

        // Loop over the quark components
        for (apfel::QuarkFlavour comp : DH.GetTagging())
          {
            // Combine perturbative contributions to the coefficient
            // functions
            apfel::Set<apfel::Operator> Ki = F2Obj.C0.at(comp);
            if (PerturbativeOrder > 0)
              Ki += ( as / apfel::FourPi ) * F2Obj.C1.at(comp);
            if (PerturbativeOrder > 1)
              Ki += pow(as / apfel::FourPi, 2) * F2Obj.C2.at(comp);

            // Convolute coefficient functions with the evolution
            // operators
            for (int j = 0; j < 13; j++)
              {
                std::map<int, apfel::Operator> gj;
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip.begin(), skip.end(), i) != skip.end()))
                    gj.insert({i, Zero});
                  else
                    gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                // Convolute distributions, combine them and return.
                Cj.at(j) += (Ki * apfel::Set<apfel::Operator> {F2Obj.ConvBasis.at(comp), gj}).Combine();
              }

            // Update total cross sections
            xsec += apfel::GetSIATotalCrossSection(PerturbativeOrder, Vs, as, aem, _Thresholds, comp);
          }

        // If the cross section is not normalised, set the total cross
        // section to one.
        if (!DH.GetNormalised())
          xsec = 1;

        // Combine coefficient functions with the total cross section
        // prefactor, including the overall prefactor. Push the same
        // resulting set of operators into the FK container as many
        // times as bins. This is not optimal but more symmetric with
        // the SIDIS case.
        for (int i = 0; i < (int) _bins.size(); i++)
          if (!_cutmask[i])
            _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
          else
            _FKt.push_back(apfel::Set<apfel::Operator>
            {(pref * apfel::GetSIATotalCrossSection(0, Vs, as, aem, _Thresholds, apfel::QuarkFlavour::TOTAL, true) / xsec ) * apfel::Set<apfel::Operator>{_cmap, Cj}});
      }
    else if (DH.GetProcess() == NangaParbat::DataHandler::Process::DIS)
      {
        // Initialise structure-function objects. This is fast enough
        // that both NC and CC can be initialised even though only one
        // of them will be used.
        const auto F2ObjNC = apfel::InitializeF2NCObjectsZM(*_g, _Thresholds);
        const auto FLObjNC = apfel::InitializeFLNCObjectsZM(*_g, _Thresholds);
        const auto F3ObjNC = apfel::InitializeF3NCObjectsZM(*_g, _Thresholds);

        const auto F2ObjCCPlus = apfel::InitializeF2CCPlusObjectsZM(*_g, _Thresholds);
        const auto FLObjCCPlus = apfel::InitializeFLCCPlusObjectsZM(*_g, _Thresholds);
        const auto F3ObjCCPlus = apfel::InitializeF3CCPlusObjectsZM(*_g, _Thresholds);

        const auto F2ObjCCMinus = apfel::InitializeF2CCMinusObjectsZM(*_g, _Thresholds);
        const auto FLObjCCMinus = apfel::InitializeFLCCMinusObjectsZM(*_g, _Thresholds);
        const auto F3ObjCCMinus = apfel::InitializeF3CCMinusObjectsZM(*_g, _Thresholds);

        const std::vector<double> flux_factor = FractureFuncFluxFactor();// HH
        // Loop on the bins
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            // Kinematics of the single bin
            const double Q = _bins[i].Qav;
            const double y = _bins[i].yav;

            const double Yp = 1 + pow(1 - y, 2);
            const double Ym = 1 - pow(1 - y, 2);

            // Get the strong coupling
            const double as = Alphas(Q);

            // Get lepton charge
            const int sign = DH.GetCharge();

            // Get evolution-operator objects
            std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Q).GetObjects();

            // Initialise container for the FK table
            std::map<int, apfel::Operator> Cj;
            for (int j = 0; j < 13; j++)
              Cj.insert({j, Zero});

            // Neutral Current cross section case
            if (_obs == NangaParbat::DataHandler::Observable::NC_red_cs)
              {
                // Electroweak charges
                const std::vector<double> Bq = apfel::ElectroWeakCharges(Q, false);
                const std::vector<double> Dq = apfel::ParityViolatingElectroWeakCharges(Q, false);

                // Get structure objects at the scale Q
                const apfel::StructureFunctionObjects F2 = F2ObjNC(Q, Bq);
                const apfel::StructureFunctionObjects FL = FLObjNC(Q, Bq);
                const apfel::StructureFunctionObjects F3 = F3ObjNC(Q, Dq);

                // Get skip vectors
                const std::vector<int> skip2 = F2.skip;
                const std::vector<int> skip3 = F3.skip;

                // Get partonic cross sections
                apfel::Set<apfel::Operator> KiF2 = F2.C0.at(0);
                apfel::Set<apfel::Operator> KiFL = FL.C0.at(0);
                apfel::Set<apfel::Operator> KiF3 = F3.C0.at(0);
                if (PerturbativeOrder > 0)
                  {
                    KiF2 += ( as / apfel::FourPi ) * F2.C1.at(0);
                    KiFL += ( as / apfel::FourPi ) * FL.C1.at(0);
                    KiF3 += ( as / apfel::FourPi ) * F3.C1.at(0);
                  }
                if (PerturbativeOrder > 1)
                  {
                    KiF2 += pow(as / apfel::FourPi, 2) * F2.C2.at(0);
                    KiFL += pow(as / apfel::FourPi, 2) * FL.C2.at(0);
                    KiF3 += pow(as / apfel::FourPi, 2) * F3.C2.at(0);
                  }
                // HH: for now multiply coefficient functions with flux factor
                KiF2 *= flux_factor[i];//HH
                KiFL *= flux_factor[i];//HH
                // Convolute coefficient functions with the evolution
                // operators for F2 and FL components
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip2.begin(), skip2.end(), i) != skip2.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += (KiF2 * apfel::Set<apfel::Operator> {F2.ConvBasis.at(0), gj}).Combine() - ( pow(y, 2) / Yp ) * (KiFL * apfel::Set<apfel::Operator> {FL.ConvBasis.at(0), gj}).Combine();
                  }

                // Convolute coefficient functions with the evolution
                // operators for the F3 component
                // for (int j = 0; j < 13; j++)
                //   {
                //     std::map<int, apfel::Operator> gj;
                //     for (int i = 0; i < 13; i++)
                //       if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip3.begin(), skip3.end(), i) != skip3.end()))
                //         gj.insert({i, Zero});
                //       else
                //         gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                //     // Convolute distributions, combine them and return.
                //     Cj.at(j) += - sign * ( Ym / Yp ) * (KiF3 * apfel::Set<apfel::Operator> {F3.ConvBasis.at(0), gj}).Combine();
                //   }
              }
            else if (_obs == NangaParbat::DataHandler::Observable::CC_red_cs)
              {
                // Get structure objects at the scale Q
                const apfel::StructureFunctionObjects F2p = F2ObjCCPlus(Q, apfel::CKM2);
                const apfel::StructureFunctionObjects FLp = FLObjCCPlus(Q, apfel::CKM2);
                const apfel::StructureFunctionObjects F3p = F3ObjCCPlus(Q, apfel::CKM2);

                const apfel::StructureFunctionObjects F2m = F2ObjCCMinus(Q, apfel::CKM2);
                const apfel::StructureFunctionObjects FLm = FLObjCCMinus(Q, apfel::CKM2);
                const apfel::StructureFunctionObjects F3m = F3ObjCCMinus(Q, apfel::CKM2);

                // Get skip vectors
                const std::vector<int> skip2p = F2p.skip;
                const std::vector<int> skip3p = F3p.skip;
                const std::vector<int> skip2m = F2m.skip;
                const std::vector<int> skip3m = F3m.skip;

                // Get partonic cross sections
                apfel::Set<apfel::Operator> KiF2p = F2p.C0.at(0);
                apfel::Set<apfel::Operator> KiFLp = FLp.C0.at(0);
                apfel::Set<apfel::Operator> KiF3p = F3p.C0.at(0);
                apfel::Set<apfel::Operator> KiF2m = F2m.C0.at(0);
                apfel::Set<apfel::Operator> KiFLm = FLm.C0.at(0);
                apfel::Set<apfel::Operator> KiF3m = F3m.C0.at(0);
                if (PerturbativeOrder > 0)
                  {
                    KiF2p += ( as / apfel::FourPi ) * F2p.C1.at(0);
                    KiFLp += ( as / apfel::FourPi ) * FLp.C1.at(0);
                    KiF3p += ( as / apfel::FourPi ) * F3p.C1.at(0);
                    KiF2m += ( as / apfel::FourPi ) * F2m.C1.at(0);
                    KiFLm += ( as / apfel::FourPi ) * FLm.C1.at(0);
                    KiF3m += ( as / apfel::FourPi ) * F3m.C1.at(0);
                  }
                if (PerturbativeOrder > 1)
                  {
                    KiF2p += pow(as / apfel::FourPi, 2) * F2p.C2.at(0);
                    KiFLp += pow(as / apfel::FourPi, 2) * FLp.C2.at(0);
                    KiF3p += pow(as / apfel::FourPi, 2) * F3p.C2.at(0);
                    KiF2m += pow(as / apfel::FourPi, 2) * F2m.C2.at(0);
                    KiFLm += pow(as / apfel::FourPi, 2) * FLm.C2.at(0);
                    KiF3m += pow(as / apfel::FourPi, 2) * F3m.C2.at(0);
                  }

                // Convolute coefficient functions with the evolution
                // operators for F2p and FLp components
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip2p.begin(), skip2p.end(), i) != skip2p.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += ( Yp / 2 ) * (KiF2p * apfel::Set<apfel::Operator> {F2p.ConvBasis.at(0), gj}).Combine()
                                - ( pow(y, 2) / 2 ) * (KiFLp * apfel::Set<apfel::Operator> {FLp.ConvBasis.at(0), gj}).Combine();
                  }

                // Now include F2m and FLm components
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip2m.begin(), skip2m.end(), i) != skip2m.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += ( sign * Yp / 2 ) * (KiF2m * apfel::Set<apfel::Operator> {F2m.ConvBasis.at(0), gj}).Combine()
                                - ( sign * pow(y, 2) / 2 ) * (KiFLm * apfel::Set<apfel::Operator> {FLm.ConvBasis.at(0), gj}).Combine();
                  }

                // Convolute coefficient functions with the evolution
                // operators for the F3p component
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip3p.begin(), skip3p.end(), i) != skip3p.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += ( - Ym / 2 ) * (KiF3p * apfel::Set<apfel::Operator> {F3p.ConvBasis.at(0), gj}).Combine();
                  }

                // Now include F3m component
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip3m.begin(), skip3m.end(), i) != skip3m.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += ( - sign * Ym / 2 ) * (KiF3m * apfel::Set<apfel::Operator> {F3m.ConvBasis.at(0), gj}).Combine();
                  }
              }

            // Combine coefficient functions with the total cross section
            // prefactor, including the overall prefactor. Push the same
            // resulting set of operators into the FK container as many
            // times as bins. This is not optimal but more symmetric with
            // the SIDIS case.
            if (!_cutmask[i])
              _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
            else
              _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, Cj});
          }
      }
    else if (DH.GetProcess() == NangaParbat::DataHandler::Process::SIDIS)
      {
        // PDF set
        const LHAPDF::PDF* PDFs = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(), config["pdfset"]["member"].as<int>());

        // Target isoscalarity
        const double iso = DH.GetTargetIsoscalarity();

        // Adjust PDFs to account for the isoscalarity
        const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
        {
          const std::map<int, double> pr = PDFs->xfxQ(x, Q);
          std::map<int, double> tg = pr;
          tg.at(1)  = iso * pr.at(1)  + ( 1 - iso ) * pr.at(2);
          tg.at(2)  = iso * pr.at(2)  + ( 1 - iso ) * pr.at(1);
          tg.at(-1) = iso * pr.at(-1) + ( 1 - iso ) * pr.at(-2);
          tg.at(-2) = iso * pr.at(-2) + ( 1 - iso ) * pr.at(-1);
          return tg;
        };

        // Rotate input PDF set into the QCD evolution basis
        const auto RotPDFs = [=] (double const& x, double const& mu) -> std::map<int, double> { return apfel::PhysToQCDEv(tPDFs(x, mu)); };

        // EW charges. Set to zero charges of the flavours that are
        // not tagged.
        std::function<std::vector<double>(double const&)> fBq = [=] (double const& Q) -> std::vector<double> { return apfel::ElectroWeakCharges(Q, false); };

        // Initialise inclusive structure functions
        const auto IF2 = BuildStructureFunctions(InitializeF2NCObjectsZM(*_g, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);
        const auto IFL = BuildStructureFunctions(InitializeFLNCObjectsZM(*_g, _Thresholds), RotPDFs, PerturbativeOrder, Alphas, fBq);

        // Inclusive cross section differential in x and Q as a
        // distribution function of Q
        const std::function<apfel::Distribution(double const&)> IncXSecQ = [&] (double const& Q) -> apfel::Distribution
        {
          // Overall Q-dependent factor of the cross section
          const double fact = 4 * M_PI * pow(Alphaem(Q), 2) / pow(Q, 3);

          // Functions that multiply F2 and FL
          const std::function<double(double const&)> func2 = [=] (double const& x) -> double{ return fact * ( 1 + pow(1 - pow(Q / Vs, 2) / x, 2) ) / x; };
          const std::function<double(double const&)> funcL = [=] (double const& x) -> double{ return - fact * pow(Q / Vs, 4) / pow(x, 3); };

          // Return cross section
          return func2 * IF2.at(0).Evaluate(Q) + funcL * IFL.at(0).Evaluate(Q);
        };

        // Tabulate total inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Distribution> TabIncXSecQ{IncXSecQ, 100, 1, 10, 3, _Thresholds};

        // Rotation Matrix from evolution to physical basis
        std::map<int, std::map<int, double>> Tqi;
        for (int q = -6; q <= 6; q++)
          {
            if (q == 0)
              continue;
            Tqi.insert({q, std::map<int, double>{}});
            for (int j = 1; j <= 12; j++)
              Tqi[q].insert({j, apfel::RotQCDEvToPhysFull[q+6][j]});
          }

        // Initialize SIDIS objects.
        const apfel::SidisObjects so = InitializeSIDIS(*_g, _Thresholds);

        // Semi-inclusive hard cross sections differential in x, Q, and z
        // as a Set<DoubleObject<Distribution, Operator>> function of Q.
        const std::function<apfel::Set<apfel::DoubleObject<apfel::Distribution, apfel::Operator>>(double const&)> Ki = [&] (double const& Q) ->
                                                                                                                       apfel::Set<apfel::DoubleObject<apfel::Distribution, apfel::Operator>>
        {
          // Strong coupling
          const double as  = Alphas(Q) / apfel::FourPi;
          const double as2 = as * as;

          // Number of active flavours
          const int nf = apfel::NF(Q, _Thresholds);

          // Get charges
          const std::vector<double> Bq = fBq(Q);

          // Overall Q-dependent factor of the cross section
          const double fact = 4 * M_PI * pow(Alphaem(Q), 2) / pow(Q, 3);

          // Functions that multiply F2 and FL
          const std::function<double(double const)> func2 = [=] (double const& x) -> double{ return fact * ( 1 + pow(1 - pow(Q / Vs, 2) / x, 2) ) / x; };
          const std::function<double(double const)> funcL = [=] (double const& x) -> double{ return - fact * pow(Q / Vs, 4) / pow(x, 3); };

          // Produce a map of distributions out of the PDFs in the
          // physical basis
          const std::map<int, apfel::Distribution> DistPDFs = apfel::DistributionMap(*_g, tPDFs, Q);

          // Initialise a map of double objects to be used to construct
          // a set
          std::map<int, apfel::DoubleObject<apfel::Distribution, apfel::Operator>> KiMap{};

          // Start with the gluon channel and construct the relevant
          // combination of PDFs
          apfel::Distribution eqfq = Bq[0] * ( DistPDFs.at(1) + DistPDFs.at(-1) );
          for (int q = 2; q <= 5; q++)
            eqfq += Bq[q-1] * ( DistPDFs.at(q) + DistPDFs.at(-q) );

          // NLO gq for F2 and FL
          apfel::DoubleObject<apfel::Distribution, apfel::Operator> D0t;
          for (auto const& t: so.C21gq.GetTerms())
            D0t.AddTerm({as * t.coefficient, func2 * ( t.object1 * eqfq ), t.object2});
          for (auto const& t: so.CL1gq.GetTerms())
            D0t.AddTerm({as * t.coefficient, funcL * ( t.object1 * eqfq ), t.object2});
          KiMap.insert({0, D0t});

          // Now run over the quark evolution basis
          for (int i = 1; i < 13; i++)
            {
              apfel::DoubleObject<apfel::Distribution, apfel::Operator> Dit;

              // Useful combination
              apfel::Distribution eqfqTqi = Bq[0] * ( DistPDFs.at(1) * Tqi.at(1).at(i) + DistPDFs.at(-1) * Tqi.at(-1).at(i) );
              for (int q = 2; q <= 5; q++)
                eqfqTqi += Bq[q-1] * ( DistPDFs.at(q) * Tqi.at(q).at(i) + DistPDFs.at(-q) * Tqi.at(-q).at(i) );

              // LO (F2 only)
              for (auto const& t: so.C20qq.GetTerms())
                Dit.AddTerm({t.coefficient, func2 * ( t.object1 * eqfqTqi ), t.object2});

              // NLO qq for F2 and FL
              for (auto const& t: so.C21qq.GetTerms())
                Dit.AddTerm({as * t.coefficient, func2 * ( t.object1 * eqfqTqi ), t.object2});
              for (auto const& t: so.CL1qq.GetTerms())
                Dit.AddTerm({as * t.coefficient, funcL * ( t.object1 * eqfqTqi ), t.object2});

              // NLO qg for F2 and FL
              double eqTqi = 0;
              for (int q = 1; q <= 5; q++)
                eqTqi += Bq[q-1] * ( Tqi.at(q).at(i) + Tqi.at(-q).at(i) );

              if (eqTqi != 0)
                {
                  for (auto const& t: so.C21qg.GetTerms())
                    Dit.AddTerm({as * eqTqi * t.coefficient, func2 * ( t.object1 * DistPDFs.at(21) ), t.object2});
                  for (auto const& t: so.CL1qg.GetTerms())
                    Dit.AddTerm({as * eqTqi * t.coefficient, funcL * ( t.object1 * DistPDFs.at(21) ), t.object2});
                }

              // NNLO qq for F2
              if (PerturbativeOrder > 1)
                for (auto const& t: so.C22qq.at(nf).GetTerms())
                  Dit.AddTerm({as2 * t.coefficient, func2 * ( t.object1 * eqfqTqi ), t.object2});

              KiMap.insert({i, Dit});
            }
          return apfel::Set<apfel::DoubleObject<apfel::Distribution, apfel::Operator>>{KiMap};
        };

        // Tabulate semi-inclusive cross sections in Q
        const apfel::TabulateObject<apfel::Set<apfel::DoubleObject<apfel::Distribution, apfel::Operator>>> TabKi{Ki, 100, 1, 10, 3, _Thresholds};

        // Pointers to the tabulated functions
        std::unique_ptr<apfel::TabulateObject<double>> TabIncQIntegrand;
        std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabSemiIncQIntegrand;

        // Keep track of the integration bounds to avoid unneeded
        // computations
        double xl = -1;
        double xu = -1;
        double xc = -1;
        double Ql = -1;
        double Qu = -1;
        double Qc = -1;

        // Integrate inclusive cross sections and store them
        for (int i = 0; i < (int) _bins.size(); i++)
          {
            double Qmin;
            double Qmax;
            if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
              {
                Qmin = std::max(sqrt(_bins[i].xmin * _bins[i].ymin) * Vs, DH.GetKinematics().var1b.first);
                Qmax = sqrt(_bins[i].xmax * _bins[i].ymax) * Vs;
              }
            else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
              {
                Qmin = _bins[i].Qmin;
                Qmax = _bins[i].Qmax;
              }
            else
              throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

            // If the point does not obey the cut, set FK table to zero and continue
            if (!_cutmask[i])
              {
                _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                continue;
              }

            // If the integration bounds in x and Q are the same, no need
            // to redo the computation.
            if ((_bins[i].Intx ? xl == _bins[i].xmin && xu == _bins[i].xmax : xc == _bins[i].xav) &&
                (_bins[i].IntQ ? Ql == Qmin && Qu == Qmax : Qc == _bins[i].Qav))
              {
                _FKt.push_back(_FKt.back());
                continue;
              }

            // Tabulate Q integrand for the inclusive cross section
            const std::function<double(double const&)> IncQIntegrand = [=] (double const& Q) -> double
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
                {
                  xbmin = std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax);
                  xbmax = std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin);
                  if (DH.GetKinematics().PSRed)
                    xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
                {
                  xbmin = _bins[i].xmin;
                  xbmax = _bins[i].xmax;
                  if (DH.GetKinematics().PSRed)
                    {
                      xbmin = std::max(xbmin, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.second);
                      xbmax = std::min(std::min(xbmax, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                    }
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");
              return (_bins[i].Intx ? TabIncXSecQ.Evaluate(Q).Integrate(xbmin, xbmax) : TabIncXSecQ.Evaluate(Q).Evaluate(_bins[i].xav));
            };

            // Tabulate Q integrand for the semi-inclusive cross section
            const std::function<apfel::Set<apfel::Operator>(double const&)> Nj = [&] (double const& Q) -> apfel::Set<apfel::Operator>
            {
              // Integration bounds in x
              double xbmin;
              double xbmax;
              if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdydz)
                {
                  xbmin = std::max(_bins[i].xmin, pow(Q / Vs, 2) / _bins[i].ymax);
                  xbmax = std::min(_bins[i].xmax, pow(Q / Vs, 2) / _bins[i].ymin);
                  if (DH.GetKinematics().PSRed)
                    xbmax = std::min(xbmax, 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                }
              else if (_obs == NangaParbat::DataHandler::Observable::dsigma_dxdQdz)
                {
                  xbmin = _bins[i].xmin;
                  xbmax = _bins[i].xmax;
                  if (DH.GetKinematics().PSRed)
                    {
                      xbmin = std::max(xbmin, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.second);
                      xbmax = std::min(std::min(xbmax, pow(Q / Vs, 2) / DH.GetKinematics().etaRange.first), 1 / ( 1 + pow(DH.GetKinematics().pTMin / Q, 2) ));
                    }
                }
              else
                throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Observable.");

              // Get Ki objects at the scale Q
              const std::map<int, apfel::DoubleObject<apfel::Distribution, apfel::Operator>> Ki = TabKi.Evaluate(Q).GetObjects();

              // Compute integral of Ki in x and construct a set
              std::map<int, apfel::Operator> IntKi;
              for (auto const& tms : Ki)
                {
                  apfel::Operator cumulant = Zero;
                  for (auto const& t : tms.second.GetTerms())
                    cumulant += t.coefficient * (_bins[i].Intx ? t.object1.Integrate(xbmin, xbmax) : t.object1.Evaluate(_bins[i].xav)) * t.object2;

                  IntKi.insert({tms.first, cumulant});
                };

              // Get evolution operator
              apfel::Set<apfel::Operator> Gammaij = TabGammaij->Evaluate(Q);

              // Intialise container for the FK table
              std::map<int, apfel::Operator> Nj;
              for (int j = 0; j < 13; j++)
                Nj.insert({j, Zero});

              // Compute the product of Ki and Gammaij and adjust the
              // convolution basis
              for (int j = 0; j < 13; j++)
                for (int i = 0; i < 13; i++)
                  if (apfel::Gkj.count({i, j}) != 0)
                    Nj.at(j) += IntKi.at(i) * Gammaij.at(apfel::Gkj.at({i, j}));

              // Return the result
              return apfel::Set<apfel::Operator> {_cmap, Nj};
            };

            // Tabulate cross section
            TabIncQIntegrand = std::unique_ptr<apfel::TabulateObject<double>> {new apfel::TabulateObject<double> {IncQIntegrand, 50, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds}};
            TabSemiIncQIntegrand = std::unique_ptr<apfel::TabulateObject<apfel::Set<apfel::Operator>>>
                                   (new apfel::TabulateObject<apfel::Set<apfel::Operator>> {Nj, 50, 0.9 * Qmin, 1.1 * Qmax, 3, _Thresholds});
            // Push back multiplicities
            if (_bins[i].IntQ)
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Integrate(Qmin, Qmax) / TabIncQIntegrand->Integrate(Qmin, Qmax)});
            else
              _FKt.push_back(apfel::Set<apfel::Operator> {pref * TabSemiIncQIntegrand->Evaluate(_bins[i].Qav) / TabIncQIntegrand->Evaluate(_bins[i].Qav)});

            xl = _bins[i].xmin;
            xu = _bins[i].xmax;
            xc = _bins[i].xav;
            Ql = Qmin;
            Qu = Qmax;
            Qc = _bins[i].Qav;
          }
      }
    else
      throw std::runtime_error("[PredictionsHandler::PredictionsHandler]: Unknown Process.");
  }

  //_________________________________________________________________________
  PredictionsHandler::PredictionsHandler(PredictionsHandler                             const& PH,
                                         std::vector<std::shared_ptr<NangaParbat::Cut>> const& cuts):
    NangaParbat::ConvolutionTable{},
    _mu0(PH._mu0),
    _Thresholds(PH._Thresholds),
    _g(PH._g),
    _obs(PH._obs),
    _bins(PH._bins),
    _qTfact(PH._qTfact),
    _cmap(PH._cmap),
    _ChargeMap(PH._ChargeMap)
  {
    // Set cuts in the mather class
    _cuts = PH._cuts;

    // Compute total cut mask as a product of single masks
    _cutmask = PH._cutmask;
    for (auto const& c : cuts)
      _cutmask *= c->GetMask();

    // Impose new cuts
    _FKt.resize(_bins.size());
    for (int i = 0; i < (int) _bins.size(); i++)
      _FKt[i] = (_cutmask[i] ? PH._FKt[i] : apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
  }

  //_________________________________________________________________________
  void PredictionsHandler::SetInputFFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc)
  {
    // Construct set of distributions
    _D = apfel::Set<apfel::Distribution> {_ChargeMap * InDistFunc(_mu0)};
  }
  //_________________________________________________________________________
  void PredictionsHandler::SetInputPDFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc)
  {
    // Construct set of distributions
    _D = apfel::Set<apfel::Distribution> {InDistFunc(_mu0)};

  }
  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::FractureFuncFluxFactor(){
  std::vector<double> res;
  double xPom = 0.0;
  double w1   = -1.191; // HH: fixed from Khanpour2019
  double w2   = 0.0;
  double w3   = 86.156;
  double w4   = 1.773;
  
  for (int i = 0; i < (int) _bins.size(); i++)
  {
    xPom = _bins[i].zav; //HH: using "z" instead of xpom, so we don't need to change DataHendler class
    res.push_back( pow(xPom, w1)*pow((1-xPom), w2)*(1 + w3 * pow(xPom, w4)) );
  };

  return res;
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const
  {
    // Initialise vector of predictions
    std::vector<double> preds(_bins.size());

    // Compute predictions by convoluting the precomputed kernels with
    // the initial-scale FFs and then perform the integration in
    // z. Finally Divide by the bin width in z.
    for (int id = 0; id < (int) _bins.size(); id++)
      if (_bins[id].Intz)
        preds[id] = (_cutmask[id] ? ((_FKt[id] * _D).Combine() * [] (double const& z) -> double{ return 1 / z; }).Integrate(_bins[id].zmin, _bins[id].zmax)
                     / ( _bins[id].zmax - _bins[id].zmin ) * _qTfact[id] : 0);
      else
        preds[id] = (_cutmask[id] ? (_FKt[id] * _D).Combine().Evaluate(_bins[id].zav) / _bins[id].zav * _qTfact[id] : 0);

    return preds;
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&,
                                                         std::function<double(double const&, double const&, double const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&, int const&)> const&,
                                                         std::function<double(double const&, double const&, double const&, int const&)> const&) const
  {
    return PredictionsHandler::GetPredictions([](double const &, double const &, double const &) -> double { return 0; });
  }
}