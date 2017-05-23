import java.util.Random;

/**
 * Created by Dan on 22/05/2017.
 */
public class Asset {

    private double[] prices;
    private double[] norms;
    private double[] weights;
    private double drift;
    private double volatility;
    private Asset untraded;

    public Asset(double drift, double volatility, double start_price, int num_to_simulate, double DT) {
        this.drift = drift;
        this.volatility = volatility;
        this.prices = new double[num_to_simulate];
        this.norms = new double[num_to_simulate];
        this.weights = new double[num_to_simulate];
        setNorms();
        setPrices(start_price, DT);
    }

    public Asset(Asset untraded, double drift, double volatility, double start_price, int num_to_simulate, double DT, double RHO) {
        this.untraded = untraded;
        this.drift = drift;
        this.volatility = volatility;
        this.prices = new double[num_to_simulate];
        this.norms = new double[num_to_simulate];
        this.weights = new double[num_to_simulate];
        setNorms(RHO);
        setPrices(start_price, DT);
    }

    private void setPrices(double start_price, double DT) {
        this.prices[0] = start_price;
        for (int i = 1; i < this.prices.length; i++) {
            this.prices[i] = this.prices[i - 1] *
                    Math.exp(
                            (this.drift - 0.5 * this.volatility * this.volatility) * (DT)
                                    + this.volatility * Math.sqrt(DT) * this.norms[i-1]
                    );
        }
    }

    private void setNorms() {
        Random random = new Random();
        for (int i = 0; i < this.norms.length; i++) {
            this.norms[i] = random.nextGaussian();
        }
    }

    private void setNorms(double RHO) {
        Random random = new Random();
        for (int i = 0; i < this.norms.length; i++) {
            this.norms[i] = RHO * this.untraded.getNorms()[i] + Math.sqrt(1 - RHO) * random.nextGaussian();
        }
    }

    public double[] getNorms() {
        return this.norms;
    }

    public double[] getPrices() {
        return this.prices;
    }

    public double getDrift() {
        return this.drift;
    }

    public double getVolatility() {
        return this.volatility;
    }

    public void setWeight(int time_period, double weight) {
        this.weights[time_period] = weight;
    }

    public void setWeightAll(double weight) {
        for (int i = 0; i < this.weights.length; i ++) {
            this.weights[i] = weight;
        }
    }

    public double getWeight(int time_period) {
        return this.weights[time_period];
    }


}
