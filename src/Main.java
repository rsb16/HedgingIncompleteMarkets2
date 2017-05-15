
/**
 * Created by Dan on 11/05/2017.
 */
public class Main {

    public static void main(String[] args) {
        int n = Integer.parseInt(args[9]);
        double profitTotal = 0;
        double profitSquared = 0;
        for (int i = 0; i < n; i++) {
            OptionPricer optionPricer = new OptionPricer(args);
            double profit = optionPricer.calculateProfitMultiHedge();
            profitTotal += profit;
            profitSquared += profit * profit;
        }
        System.out.println("E(X) = " + profitTotal / n);
        System.out.println("VAR(X) = " + ((profitSquared / n) - (profitTotal / n) * (profitTotal / n)));
    }

}
