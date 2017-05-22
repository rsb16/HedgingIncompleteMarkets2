import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.io.*;
import java.util.*;

/**
 * Created by Dan on 11/05/2017.
 */
public class OptionPricer {

    Map<Integer, Asset> assets = new HashMap<>();
    private double muCorrelated;
    private double sigmaCorrelated;
    private double muUntraded;
    private double sigmaUntraded;
    private double startPriceCorrelated;
    private double startPriceUntraded;
    private int numDays;
    private int hedgesPerDay;
    private double[] untradedPrices;
    private double[] correlatedPrices;
    private double[] dzc;
    private double[] dze;
    private static double[] NORMS1;
    private static double[] NORMS2;
    private static double DT;
    private static double INTEREST = 0.05;
    private static double STRIKE = 300;
    private static double RHO = 0.8;
    private double w;
    private double bec;
    private boolean hedge;
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
        for (int i = 0; i < Integer.parseInt(args[1]); i++) {
            this.assets.put(i, new Asset(
                    Double.parseDouble(prop.getProperty("correlated_mu_" + i)),
                    Double.parseDouble(prop.getProperty("correlated_vol_" + i)),
                    Double.parseDouble(prop.getProperty("start_price_correlated_" + i))
            ));
        }
        //this.muCorrelated = Double.parseDouble(prop.getProperty("correlated_mu"));
        //this.sigmaCorrelated = Double.parseDouble(prop.getProperty("correlated_vol"));
        this.muUntraded = Double.parseDouble(prop.getProperty("untraded_mu"));
        this.sigmaUntraded = Double.parseDouble(prop.getProperty("untraded_vol"));
        this.startPriceUntraded = Double.parseDouble(prop.getProperty("start_price_untraded"));
        //this.startPriceCorrelated = Double.parseDouble(prop.getProperty("start_price_correlated"));
        this.numDays = Integer.parseInt(prop.getProperty("num_days"));
        this.hedgesPerDay = Integer.parseInt(prop.getProperty("hedges_per_day"));
        this.T = this.numDays / 365.0;
        DT = (this.T) / (this.numDays * this.hedgesPerDay);
        NORMS1 = new double[this.hedgesPerDay * this.numDays];
        NORMS2 = new double[this.hedgesPerDay * this.numDays];
        try {
            makeRandomNormals();
        } catch (Exception e) {
            System.out.println("here");
        }
        setUntradedPrices();
        setCorrelatedPrices();
        this.bec = bec();
        this.w = this.muUntraded - this.bec*(this.muCorrelated - INTEREST);
        //printPrices(n);
    }

    public double calculateOptionPrice(int day) {
        double d1 = (Math.log(this.untradedPrices[day] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - day * DT))
                / (this.sigmaUntraded * Math.sqrt(this.T - day * DT));
        double d2 = (Math.log(this.untradedPrices[day] / STRIKE) + (this.w - (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - day * DT))
                / (this.sigmaUntraded * Math.sqrt(this.T - day * DT));
        NormalDistribution distribution = new NormalDistribution(0, 1);
        double V = this.startPriceUntraded * Math.exp((w - INTEREST) * (this.T - day * DT)) * distribution.cumulativeProbability(d1)
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
        return calculateOptionPrice(0) * Math.exp(INTEREST * this.T) - Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE);
    }

    public double calculateProfitSingleHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset at time t = 0.
        //We calculate the final profit.
        double d1 = (Math.log(this.untradedPrices[0] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - 0 * DT))
                / (this.sigmaUntraded * Math.sqrt(this.T - 0 * DT));
        NormalDistribution norm = new NormalDistribution(0, 1);
        double phi = this.startPriceUntraded * Math.exp((this.w - INTEREST) * (this.T - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double G = this.calculateOptionPrice(0);
        double riskFree = G - phi;
        double moneyFromCorrelated = (phi / this.correlatedPrices[0]) * this.correlatedPrices[this.correlatedPrices.length - 1];
        return -1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE) + moneyFromCorrelated + riskFree * Math.exp(INTEREST * this.T);
    }

    public double calculateProfitMultiHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset.
        //We calculate the profit per time step.
        double profitFromCorrelated = 0;
        double costOfBorrowing = 0;
        double G = this.calculateOptionPrice(0);
        NormalDistribution norm = new NormalDistribution(0, 1);
        double d1 = (Math.log(this.untradedPrices[0] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - 0 * DT))
                / (this.sigmaUntraded * Math.sqrt(this.T - 0 * DT));
        double phiPrev = this.startPriceUntraded * Math.exp((this.w - INTEREST) * (this.T - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double riskFree = 0;
        for (int i = 1; i < this.hedgesPerDay * this.numDays; i++) {
            //Get the ith d1.
            d1 = (Math.log(this.untradedPrices[i] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (this.T - i * DT))
                    / (this.sigmaUntraded * Math.sqrt(this.T - i * DT));
            //Get the ith phi.
            double phi = this.untradedPrices[i] * Math.exp((this.w - INTEREST) * (this.T - i * DT)) * norm.cumulativeProbability(d1) * this.bec;
            double dG = ((G - phi) * INTEREST + phi * this.muCorrelated) * DT + phi * this.sigmaCorrelated * this.dzc[i];
            riskFree = G - phi;
            G += dG;
            profitFromCorrelated += (((phiPrev / this.correlatedPrices[i - 1]) * this.correlatedPrices[i]) - phiPrev);
            //System.out.println("phi: " + phi);
            //System.out.println("Corr[" + i + "]: " + this.correlatedPrices[i]);
            //System.out.println("phiPrev: " + phiPrev);
            //System.out.println("Corr[" + (i - 1) + "]: " + this.correlatedPrices[i - 1]);
            //System.out.println("Prof: " + (((phiPrev / this.correlatedPrices[i - 1]) * this.correlatedPrices[i]) - phiPrev));
            //System.out.println();
            costOfBorrowing += (riskFree * INTEREST * DT);
            phiPrev = phi;
        }
//        System.out.println("V: " + calculateOptionPrice(0));
//        System.out.println("payout: " + -Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE));
//        System.out.println("prof: " + profitFromCorrelated);
//        System.out.println("Cost: " + costOfBorrowing);
//        System.out.println("G: " + G);
        double profit = - (1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE)) + profitFromCorrelated + calculateOptionPrice(0) + costOfBorrowing;
//        System.out.println(profit);
//        System.out.println();
        return profit;
    }

    private void makeRandomNormals() {
        Random random = new Random();
            for (int i = 0; i < this.numDays * this.hedgesPerDay; i ++) {
                NORMS1[i] = random.nextGaussian();
                NORMS2[i] = RHO * NORMS1[i] + Math.sqrt(1 - RHO) * random.nextGaussian();
            }
    }


    private void setUntradedPrices() {
        this.untradedPrices = new double[this.numDays * this.hedgesPerDay];
        this.untradedPrices[0] = this.startPriceUntraded;
        for (int i = 1; i < this.untradedPrices.length; i++) {
            this.untradedPrices[i] = this.untradedPrices[i - 1] *
                    Math.exp(
                            (this.muUntraded - 0.5 * this.sigmaUntraded * this.sigmaUntraded) * (DT)
                                    + this.sigmaUntraded * Math.sqrt(DT) * NORMS1[i-1]
                    );
        }
    }

    private void setCorrelatedPrices() {
        this.correlatedPrices = new double[this.numDays * this.hedgesPerDay];
        this.correlatedPrices[0] = this.startPriceCorrelated;
        for (int i = 1; i < this.correlatedPrices.length; i++) {
            this.correlatedPrices[i] = this.correlatedPrices[i - 1] *
                    Math.exp(
                            (this.muCorrelated - 0.5 * this.sigmaCorrelated * this.sigmaCorrelated) * (DT)
                                    + this.sigmaCorrelated * Math.sqrt(DT) * NORMS2[i-1]
                    );
        }
    }

    public void printPrices(int n) {
        try {
            PrintWriter untradedWriter = new PrintWriter("untraded-prices" + n + ".txt", "UTF-8");
            PrintWriter correlatedWriter = new PrintWriter("correlated-prices" + n + ".txt", "UTF-8");
            for (int i = 0; i < this.numDays * this.hedgesPerDay; i++) {
                untradedWriter.print(i + " ");
                untradedWriter.println(this.untradedPrices[i] + " ");
                correlatedWriter.print(i + " ");
                correlatedWriter.println(this.correlatedPrices[i] + " ");
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
        for (int i = 1; i < this.numDays * this.hedgesPerDay; i++) {
            this.dze[i] = (((this.untradedPrices[i] - this.untradedPrices[i-1]) - this.muUntraded * this.untradedPrices[i] * DT)/ (this.sigmaUntraded * this.untradedPrices[i]));
            this.dzc[i] = (((this.correlatedPrices[i] - this.correlatedPrices[i-1]) - this.muCorrelated * this.correlatedPrices[i] * DT)/ (this.sigmaCorrelated * this.correlatedPrices[i]));
        }
        Covariance cov = new Covariance();
        double pec = cov.covariance(this.dze, this.dzc) / StatUtils.variance(this.dzc);
        return pec * this.sigmaUntraded / this.sigmaCorrelated;
    }
}
