import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Properties;

/**
 * Created by Dan on 11/05/2017.
 */
public class Main {

    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        double profitTotal = 0;
        double profitSquared = 0;
        try {
            PrintWriter writer = new PrintWriter("profitshedging10x.txt", "UTF-8");
            FileWriter untraded_writer = new FileWriter("asset0.txt", true);
            FileWriter asset_1_writer = new FileWriter("asset1.txt", true);
            FileWriter asset_2_writer = new FileWriter("asset2.txt", true);
            for (int i = 0; i < n; i++) {
                OptionPricer optionPricer = new OptionPricer(args, i);
                double profit = optionPricer.calculateProfitMultiHedge();
                for (double price : optionPricer.untraded.getPrices()) {
                    untraded_writer.write(price + "\n");
                }
                for (double price : optionPricer.assets.get(0).getPrices()) {
                    asset_1_writer.write(price + "\n");
                }
                for (double price : optionPricer.assets.get(1).getPrices()) {
                    asset_2_writer.write(price + "\n");
                }
                writer.print(i + ",");
                profitTotal += profit;
                writer.println(profitTotal / (i + 1.0));
                profitSquared += profit * profit;
            }
            writer.close();
            untraded_writer.close();
            asset_1_writer.close();
            asset_2_writer.close();
        } catch (Exception e) {
            System.out.println(e);
        }
        System.out.println("E(X) = " + profitTotal / n);
        System.out.println("VAR(X) = " + ((profitSquared / n) - (profitTotal / n) * (profitTotal / n)));
    }

}
