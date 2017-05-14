import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

/**
 * Created by Dan on 11/05/2017.
 */
public class OptionPricer {

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
    private static double[] NORMS;
    private static double DT;
    private static double INTEREST = 0.05;
    private static double STRIKE = 300;
    private double w;
    private double bec;
    private boolean hedge;

    public OptionPricer(String[] args) {
        this.muCorrelated = Double.parseDouble(args[0]);
        this.sigmaCorrelated = Double.parseDouble(args[1]);
        this.muUntraded = Double.parseDouble(args[2]);
        this.sigmaUntraded = Double.parseDouble(args[3]);
        this.startPriceUntraded = Double.parseDouble(args[4]);
        this.startPriceCorrelated = Double.parseDouble(args[5]);
        this.numDays = Integer.parseInt(args[6]);
        this.hedgesPerDay = Integer.parseInt(args[7]);
        this.hedge = Boolean.parseBoolean(args[8].split("=")[1]);
        DT = 1.0 / (this.numDays * this.hedgesPerDay);

        NORMS = new double[this.hedgesPerDay * this.numDays];
        try {
            makeRandomNormals();
            setRandoms();
        } catch (Exception e) {

        }
        setUntradedPrices();
        setCorrelatedPrices();
        this.bec = bec();
        this.w = this.muUntraded - this.bec*(this.muCorrelated - INTEREST);
        //printPrices();
    }

    public double calculateOptionPrice(int day) {
        double d1 = (Math.log10(this.untradedPrices[day] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (1 - day * DT))
                / (this.sigmaUntraded * Math.sqrt(1 - day * DT));
        double d2 = (Math.log10(this.untradedPrices[day] / STRIKE) + (this.w - (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (1 - day * DT))
                / (this.sigmaUntraded * Math.sqrt(1 - day * DT));
        NormalDistribution distribution = new NormalDistribution(0, 1);
        double V = this.startPriceUntraded * Math.exp((w - INTEREST) * (1 - day * DT)) * distribution.cumulativeProbability(d1)
                - STRIKE * Math.exp(-1 * INTEREST * (1 - day * DT)) * distribution.cumulativeProbability(d2);
        //System.out.println("V: " + V);
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
        return calculateOptionPrice(0) - Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE);
    }

    public double calculateProfitSingleHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset at time t = 0.
        //We calculate the final profit.
        double d1 = (Math.log10(this.untradedPrices[0] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (1 - 0 * DT))
                / (this.sigmaUntraded * Math.sqrt(1 - 0 * DT));
        NormalDistribution norm = new NormalDistribution(0, 1);
        double phi = this.startPriceUntraded * Math.exp((this.w - INTEREST) * (1 - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double G = this.calculateOptionPrice(0);
        double riskFree = G - phi;
        double moneyFromCorrelated = (phi / this.correlatedPrices[0]) * this.correlatedPrices[this.correlatedPrices.length - 1];
        double costOfBorrowing = riskFree * INTEREST * (numDays / 365.0);
        System.out.println(-1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE) + moneyFromCorrelated + riskFree * Math.exp(INTEREST * (numDays / 365.0)));
        return -1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE) + moneyFromCorrelated + riskFree * Math.exp(INTEREST * (numDays / 365.0));
    }

    public double calculateProfitMultiHedge() {
        //We take (G - phi) and invest it in the risk free.
        //We take phi and invest it in the correlated asset.
        //We calculate the profit per time step.
        double profitFromCorrelated = 0;
        double costOfBorrowing = 0;
        double G = this.calculateOptionPrice(0);
        NormalDistribution norm = new NormalDistribution(0, 1);
        double d1 = (Math.log10(this.untradedPrices[0] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (1 - 0 * DT))
                / (this.sigmaUntraded * Math.sqrt(1 - 0 * DT));
        double phiPrev = this.startPriceUntraded * Math.exp((this.w - INTEREST) * (1 - 0 * DT)) * norm.cumulativeProbability(d1) * this.bec;
        double riskFree = 0;
        for (int i = 1; i < this.hedgesPerDay * this.numDays; i++) {
            //Get the ith d1.
            d1 = (Math.log10(this.untradedPrices[i] / STRIKE) + (this.w + (0.5) * this.sigmaUntraded * this.sigmaUntraded) * (1 - i * DT))
                    / (this.sigmaUntraded * Math.sqrt(1 - i * DT));
            //Get the ith phi.
            double phi = this.untradedPrices[i] * Math.exp((this.w - INTEREST) * (1 - i * DT)) * norm.cumulativeProbability(d1) * this.bec;
            double dG = ((G - phi) * INTEREST + phi * this.muCorrelated) * DT + phi * this.sigmaCorrelated * this.dzc[i];
            riskFree = G - phi;
            G += dG;
            profitFromCorrelated += (((phi - phiPrev) / this.correlatedPrices[i-1]) * this.correlatedPrices[i]);
            costOfBorrowing += (riskFree * INTEREST * DT);
            phiPrev = phi;
        }
        System.out.println("payout: " + -Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE));
        System.out.println("profCorr: " + profitFromCorrelated);
        System.out.println("riskFree: " + riskFree * Math.exp(INTEREST));
        System.out.println("phiPrev: " + phiPrev);
        double profit = - (1 * Math.max(0, this.untradedPrices[this.untradedPrices.length - 1] - STRIKE)) + riskFree * Math.exp(INTEREST) + phiPrev;
        System.out.println(profit);
        System.out.println();
        return profit;
    }

    private void makeRandomNormals() {
        Random random = new Random();
        try{
            PrintWriter writer = new PrintWriter("randoms.txt", "UTF-8");
            for (int i = 0; i < this.numDays * this.numDays; i ++) {
                writer.println(random.nextGaussian());
            }
            writer.close();
        } catch (IOException e) {
            // do something
        }
    }

    private void setRandoms() throws Exception {
        BufferedReader br = new BufferedReader(new FileReader("/Users/Dan/IdeaProjects/TestingLuenberger2/randoms.txt"));
        try {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < this.numDays * this.hedgesPerDay; i++) {
                String line = br.readLine();
                NORMS[i] = Double.parseDouble(line);
            }

        }
        finally {
            br.close();
        }
    }

    private void setUntradedPrices() {
        this.untradedPrices = new double[this.numDays * this.hedgesPerDay];
        this.untradedPrices[0] = this.startPriceUntraded;
        for (int i = 1; i < this.untradedPrices.length; i++) {
            this.untradedPrices[i] = this.untradedPrices[i - 1] *
                    Math.exp(
                            (this.muUntraded - 0.5 * this.sigmaUntraded * this.sigmaUntraded) * DT
                                    + this.sigmaUntraded * Math.sqrt(DT) * NORMS[i-1]
                    );
        }
    }

    private void setCorrelatedPrices() {
        this.correlatedPrices = new double[this.numDays * this.hedgesPerDay];
        this.correlatedPrices[0] = this.startPriceCorrelated;
        for (int i = 1; i < this.correlatedPrices.length; i++) {
            this.correlatedPrices[i] = this.correlatedPrices[i - 1] *
                    Math.exp(
                            (this.muCorrelated - 0.5 * this.sigmaCorrelated * this.sigmaCorrelated) * DT
                                    + this.sigmaCorrelated * Math.sqrt(DT) * NORMS[i-1]
                    );
        }
    }

    public void printPrices() {
        for (int i = 0; i < this.numDays * this.hedgesPerDay; i++) {
            System.out.println("Xe(" + i + ")" + " = " + this.untradedPrices[i]);
            System.out.println("Xc(" + i + ")" + " = " + this.correlatedPrices[i]);
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
