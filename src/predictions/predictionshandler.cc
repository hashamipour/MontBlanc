/*
 * Authors: Chiara Bissolotti: chiara.bissolotti01@ateneopv.it
 *          Valerio Bertone: valerio.bertone@cern.ch
 */

#include "NavyPier/predictionshandler.h"

#include <LHAPDF/LHAPDF.h>
#include <apfel/zeromasscoefficientfunctionsunp_tl.h>

#include <numeric>

namespace NavyPier
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
    _cmap(apfel::DiagonalBasis{13})
  {
    // Set silent mode for both apfel=+ and LHAPDF;
    apfel::SetVerbosityLevel(0);
    LHAPDF::setVerbosity(0);

    // Perturbative order
    const int PerturbativeOrder = config["perturbative order"].as<int>();

    // Alpha_s
    const apfel::TabulateObject<double> TabAlphas{*(new apfel::AlphaQCD
      {config["alphas"]["aref"].as<double>(), config["alphas"]["Qref"].as<double>(), _Thresholds, PerturbativeOrder}), 100, 0.9, 1001, 3};
    const auto Alphas = [&] (double const& mu) -> double{ return TabAlphas.Evaluate(mu); };

    // Initialize QCD space-like evolution operators and tabulate them
    const std::unique_ptr<const apfel::TabulateObject<apfel::Set<apfel::Operator>>> TabGammaij{new const apfel::TabulateObject<apfel::Set<apfel::Operator>>
      {*(BuildDglap(InitializeDglapObjectsQCD(*_g, _Thresholds, true), _mu0, PerturbativeOrder, Alphas)), 100, 1, 300, 3}};

    // Zero operator
    const apfel::Operator Zero{*_g, apfel::Null{}};

    // Identity operator
    const apfel::Operator Id{*_g, apfel::Identity{}};

    // Set cuts in the mother class
    this->_cuts = cuts;

    // Compute total cut mask as a product of single masks
    _cutmask.resize(_bins.size(), true);
    for (auto const& c : _cuts)
      _cutmask *= c->GetMask();

    // DIS reduced cross section
    if (DH.GetProcess() == NangaParbat::DataHandler::Process::DIS)
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
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0 || (std::find(skip3.begin(), skip3.end(), i) != skip3.end()))
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += - sign * ( Ym / Yp ) * (KiF3 * apfel::Set<apfel::Operator> {F3.ConvBasis.at(0), gj}).Combine();
                  }
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
    else if (DH.GetProcess() ==  NangaParbat::DataHandler::Process::Sum_rules)
      {
        if (_obs == NangaParbat::DataHandler::Observable::aSigma ||
            _obs == NangaParbat::DataHandler::Observable::aV     ||
            _obs == NangaParbat::DataHandler::Observable::aV3    ||
            _obs == NangaParbat::DataHandler::Observable::aV8)
          {
            for (int i = 0; i < (int) _bins.size(); i++)
              {
                // Get virtuality
                const double Q = _bins[i].Qav;

                // Get evolution-operator objects
                std::map<int, apfel::Operator> Gammaij = TabGammaij->Evaluate(Q).GetObjects();

                // Fill the coefficients Ki with Id for the selected distribution
                // and zero otherwise
                std::map<int, apfel::Operator> Ki;
                for (int j = 0; j < 13; j++)
                  if (DH.GetObservable() == NangaParbat::DataHandler::Observable::aSigma && (j == 0 || j == 1))
                    Ki.insert({j, Id});
                  else if (DH.GetObservable() == NangaParbat::DataHandler::Observable::aV && j == 2)
                    Ki.insert({j, Id});
                  else if (DH.GetObservable() == NangaParbat::DataHandler::Observable::aV3 && j == 4)
                    Ki.insert({j, Id});
                  else if (DH.GetObservable() == NangaParbat::DataHandler::Observable::aV8 && j == 6)
                    Ki.insert({j, Id});
                  else
                    Ki.insert({j, Zero});

                // Intialise container for the FK table
                std::map<int, apfel::Operator> Cj;
                for (int j = 0; j < 13; j++)
                  Cj.insert({j, Zero});

                // Convolute coefficient functions with the evolution
                // operators
                for (int j = 0; j < 13; j++)
                  {
                    std::map<int, apfel::Operator> gj;
                    for (int i = 0; i < 13; i++)
                      if (apfel::Gkj.count({i, j}) == 0)
                        gj.insert({i, Zero});
                      else
                        gj.insert({i, Gammaij.at(apfel::Gkj.at({i, j}))});

                    // Convolute distributions, combine them and return.
                    Cj.at(j) += (apfel::Set<apfel::Operator> {_cmap, Ki} * apfel::Set<apfel::Operator> {_cmap, gj}).Combine();
                  }

                // Put together FK tables
                if (!_cutmask[i])
                  _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, std::map<int, apfel::Operator>{}});
                else
                  _FKt.push_back(apfel::Set<apfel::Operator> {_cmap, Cj});
              }
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
    _cmap(PH._cmap)
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
  void PredictionsHandler::SetInputPDFs(std::function<apfel::Set<apfel::Distribution>(double const&)> const& InDistFunc)
  {
    // Construct set of distributions
    _D = apfel::Set<apfel::Distribution> {InDistFunc(_mu0)};

  }

  //_________________________________________________________________________
  std::vector<double> PredictionsHandler::GetPredictions(std::function<double(double const&, double const&, double const&)> const&) const
  {
    // Initialise vector of predictions
    std::vector<double> preds(_bins.size());

    // Compute predictions by convoluting the precomputed kernels with
    // the initial-scale PDFs.
    for (int id = 0; id < (int) _bins.size(); id++)
      if (_bins[id].Intx)
        {
          if (_obs == NangaParbat::DataHandler::Observable::aV  ||
              _obs == NangaParbat::DataHandler::Observable::aV3 ||
              _obs == NangaParbat::DataHandler::Observable::aV8)
            preds[id] = (_cutmask[id] ? ((_FKt[id] * _D).Combine() * [] (double const& x) -> double{ return 1 / x; }).Integrate(_bins[id].xmin, _bins[id].xmax) : 0);
          else
            preds[id] = (_cutmask[id] ? ((_FKt[id] * _D).Combine()).Integrate(_bins[id].xmin, _bins[id].xmax) : 0);
        }
      else
        preds[id] = (_cutmask[id] ? (_FKt[id] * _D).Combine().Evaluate(_bins[id].xav) : 0);

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
