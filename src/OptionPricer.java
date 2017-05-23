import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.io.*;
import java.util.*;

/**
 * Created by Dan on 11/05/2017.
 */
public class OptionPricer {

    private Asset untraded;
    ArrayList<Asset> assets = new ArrayList<>();
    private int numDays;
    private int hedgesPerDay;
    private double[] dzc;
    private double[] dze;
    private static double DT;
    private static double INTEREST = 0.05;
    private static double STRIKE = 250;
    private double w;
    private double bec;
    private double T;

    public OptionPricer(String[] args, int n) {
        Properties prop = new Properties();
        try {
            FileInputStream is = new FileInputStream("Resources.properties");
            prop.load(is);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.numDays = Integer.parseInt(prop.getProperty("num_days"));
        this.hedgesPerDay = Integer.parseInt(prop.getProperty("hedges_per_day"));
        this.T = this.numDays / 365.0;
        DT = (this.T) / (this.numDays * this.hedgesPerDay);
        this.untraded = new Asset(
                Double.parseDouble(prop.getProperty("untraded_mu")),
                Double.parseDouble(prop.getProperty("untraded_vol")),
                Double.parseDouble(prop.getProperty("start_price_untraded")),
                this.numDays * this.hedgesPerDay,
                DT
        );
        for (int i = 1; i < Integer.parseInt(args[1]); i++) {
            this.assets.add(new Asset(
                    this.untraded,
                    Double.parseDouble(prop.getProperty("correlated_mu_" + i)),
                    Double.parseDouble(prop.getProperty("correlated_vol_" + i)),
                    Double.parseDouble(prop.getProperty("start_price_correlated_" + i)),
                    this.numDays * this.hedgesPerDay,
                    DT,
                    Double.parseDouble(prop.getProperty("RHO_" + i))
            ));
        }
        this.untraded.setWeightAll(1);
        setWeightsAll(); // should be replaced
        this.bec = bec();
        //this.w = this.muUntraded - this.bec*(this.muCorrelated - INTEREST);
        //printPrices(n);
    }

    public double calculateOptionPrice(int day) {
        //double d1 = (Math.log(this.untradedPrices[day] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - day * DT))
        //        / (this.sigmaUntraded * Math.sqrt(this.T - day * DT));
        //double d2 = (Math.log(this.untradedPrices[day] / STRIKE) + (this.w - (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - day * DT))
        //        / (this.sigmaUntraded * Math.sqrt(this.T - day * DT));
        double d1 = (Math.log(this.untraded.getPrices()[day] / STRIKE) + (this.w + (0.5) * this.untraded.getVolatility() * this.untraded.getVolatility()) * (this.T - day * DT))
                  / (this.untraded.getVolatility() * Math.sqrt(this.T - day * DT));
        double d2 = (Math.log(this.untraded.getPrices()[day] / STRIKE) + (this.w - (0.5) * this.untraded.getVolatility() * this.untraded.getVolatility()) * (this.T - day * DT))
                / (this.untraded.getVolatility() * Math.sqrt(this.T - day * DT));
        NormalDistribution distribution = new NormalDistribution(0, 1);
        double V = this.untraded.getPrices()[0] * Math.exp((this.w - INTEREST) * (this.T - day * DT)) * distribution.cumulativeProbability(d1)
                - STRIKE * Math.exp(-1 * INTEREST * (this.T - day * DT)) * distribution.cumulativeProbability(d2);
        return V;
    }

    public double calculateProfitNoHedge() {
        //We always get the optionPrice and have to pay out:
        //0 if the option expires out of the money
        //Final price - Strike if the option expires in the money.
        //Example 1:
        //Option value : 15
        //option expires at 250 with strike 200.
        //profit = 15 + (200 - 250) = -35.
        //Example 2:
        //Option value : 15
        //option expires at 180 with strike 200.
        //profit = 15 + 0 = 15.
        this.setWeightsAll();
        //return calculateOptionPrice(0) * Math.exp(INTEREST * this.T) - Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE);
        return 10;
    }

    public double calculateProfitSingleHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset at time t = 0.
        //We calculate the final profit.
        this.setWeightsAll();
        this.w = this.untraded.getDrift() - this.bec * (this.calculatePortfolioMu(0) - INTEREST);
        double d1 = (Math.log(this.untraded.getPrices()[0] / STRIKE) + (this.w + (0.5) * this.untraded.getVolatility() * this.untraded.getVolatility()) * (this.T - 0 * DT))
                / (this.untraded.getVolatility() * Math.sqrt(this.T - 0 * DT));
        NormalDistribution norm = new NormalDistribution(0, 1);
        double phi = this.untraded.getPrices()[0] * Math.exp((this.w - INTEREST) * (this.T - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double G = this.calculateOptionPrice(0);
        double riskFree = G - phi;
        //double moneyFromCorrelated = (phi / this.correlatedPrices[0]) * this.correlatedPrices[this.correlatedPrices.length - 1];
        double moneyFromCorrelated = (phi / calculatePortfolioValue(0)) * calculatePortfolioValue(this.hedgesPerDay * this.numDays - 1);
        //return -1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE) + moneyFromCorrelated + riskFree * Math.exp(INTEREST * this.T);
        return -1 * Math.max(0, this.untraded.getPrices()[this.untraded.getPrices().length - 1] - STRIKE) + moneyFromCorrelated + riskFree * Math.exp(INTEREST * this.T);
    }

    public double calculateProfitMultiHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset.
        //We calculate the profit per time step.
        double profitFromCorrelated = 0;
        double costOfBorrowing = 0;
        double G = this.calculateOptionPrice(0);
        this.w = this.untraded.getDrift() - this.bec * (this.calculatePortfolioMu(0) - INTEREST);
        NormalDistribution norm = new NormalDistribution(0, 1);
        double d1 = (Math.log(this.untraded.getPrices()[0] / STRIKE) + (this.w + (0.5) * this.untraded.getVolatility() * this.untraded.getVolatility()) * (this.T - 0 * DT))
                / (this.untraded.getVolatility() * Math.sqrt(this.T - 0 * DT));
        double phiPrev = this.untraded.getPrices()[0] * Math.exp((this.w - INTEREST) * (this.T - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double riskFree = 0;
        for (int i = 1; i < this.hedgesPerDay * this.numDays; i++) {
            //Get the ith d1.
            this.w = this.untraded.getDrift() - this.bec * (this.calculatePortfolioMu(i) - INTEREST);
            d1 = (Math.log(this.untraded.getPrices()[i] / STRIKE) + (this.w + (0.5) * this.untraded.getVolatility() * this.untraded.getVolatility()) * (this.T - i * DT))
                    / (this.untraded.getVolatility() * Math.sqrt(this.T - i * DT));
            //Get the ith phi.
            double phi = this.untraded.getPrices()[i] * Math.exp((this.w - INTEREST) * (this.T - i * DT)) * norm.cumulativeProbability(d1) * this.bec;
            double dG = ((G - phi) * INTEREST + phi * this.calculatePortfolioMu(i)) * DT + phi * this.calculatePortfolioSigma(i) * this.dzc[i];
            riskFree = G - phi;
            G += dG;
            profitFromCorrelated += (((phiPrev / this.calculatePortfolioValue(i - 1)) * this.calculatePortfolioValue(i)) - phiPrev);
            //System.out.println("phi: " + phi);
            //System.out.println("Corr[" + i + "]: " + this.correlatedPrices[i]);
            //System.out.println("phiPrev: " + phiPrev);
            //System.out.println("Corr[" + (i - 1) + "]: " + this.correlatedPrices[i - 1]);
            //System.out.println("Prof: " + (((phiPrev / this.correlatedPrices[i - 1]) * this.correlatedPrices[i]) - phiPrev));
            //System.out.println();
            System.out.println("V: " + calculateOptionPrice(i));
            costOfBorrowing += (riskFree * INTEREST * DT);
            phiPrev = phi;
        }
        System.out.println("V: " + calculateOptionPrice(0));
        System.out.println("payout: " + -Math.max(0, this.untraded.getPrices()[this.untraded.getPrices().length - 1] - STRIKE));
        System.out.println("prof: " + profitFromCorrelated);
        System.out.println("Cost: " + costOfBorrowing);
        System.out.println("G: " + G);
        double profit = - (1 * Math.max(0, this.untraded.getPrices()[this.untraded.getPrices().length - 1] - STRIKE)) + profitFromCorrelated + calculateOptionPrice(0) + costOfBorrowing;
        System.out.println(profit);
        System.out.println();
        return profit;
    }


    public void printPrices(int n) {
        try {
            PrintWriter untradedWriter = new PrintWriter("untraded-prices" + n + ".txt", "UTF-8");
            PrintWriter correlatedWriter = new PrintWriter("correlated-prices" + n + ".txt", "UTF-8");
            for (int i = 0; i < this.numDays * this.hedgesPerDay; i++) {
                untradedWriter.print(i + " ");
                //untradedWriter.println(this.untradedPrices[i] + " ");
                correlatedWriter.print(i + " ");
                //correlatedWriter.println(this.correlatedPrices[i] + " ");
            }
            untradedWriter.close();
            correlatedWriter.close();
        } catch (Exception e) {
            System.out.println("Error writing to file.");
        }

    }

    public double bec() {
        this.dze = new double[this.numDays * this.hedgesPerDay];
        this.dzc = new double[this.numDays * this.hedgesPerDay];
        double portfolioValuePrev = calculatePortfolioValue(0);
        for (int i = 1; i < this.numDays * this.hedgesPerDay; i++) {
            this.dze[i] = (((this.untraded.getPrices()[i] - this.untraded.getPrices()[i-1]) - this.untraded.getDrift() * this.untraded.getPrices()[i] * DT)/ (this.untraded.getVolatility() * this.untraded.getPrices()[i]));
            this.dzc[i] = (((calculatePortfolioValue(i) - portfolioValuePrev) - this.calculatePortfolioMu(i) * this.calculatePortfolioValue(i) * DT)/ (this.calculatePortfolioSigma(i) * this.calculatePortfolioValue(i)));
            portfolioValuePrev = calculatePortfolioValue(i);
        }
        Covariance cov = new Covariance();
        double pec = cov.covariance(this.dze, this.dzc) / StatUtils.variance(this.dzc);
        return pec * this.untraded.getVolatility() / this.calculatePortfolioSigma(this.dze.length - 1);
    }

    private double calculatePortfolioMu(int i) {
        double mu = 0;
        for (Asset asset : this.assets) {
            // don't include the untraded.
            mu += asset.getWeight(i) * asset.getDrift();
        }
        return mu;
    }

    private double calculatePortfolioSigma(int i) {
        double sigma = 0;
        for (Asset asset : this.assets) {
            // don't include the untraded.
            sigma += asset.getWeight(i) * asset.getVolatility();
        }
        return sigma;
    }

    private void setWeights(int time_period) {
        // check it sets to 1 when two assets etc.
        int num_assets = this.assets.size();
        for (Asset asset : this.assets) {
            asset.setWeightAll(1 / (num_assets - 1.0));
        }
    }

    private void setWeightsAll() {
        // check it sets to 1 when two assets etc.
        for (Asset asset : this.assets) {
            asset.setWeightAll(1 / this.assets.size());
        }
    }

    private double calculateWeight() {
        return 0;
    }

    private double calculatePortfolioValue(int i) {
        double value = 0;
        for (Asset asset : this.assets) {
            value += asset.getWeight(i) * asset.getPrices()[i];
        }
        return value;
    }
}
