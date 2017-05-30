for i in {0..100}
do
  python markow.py 5 0.1 >> Profits_5_0.1.txt
  python markow.py 50 0.1 >> Profits_50_0.1.txt
  python markow.py 500 0.1 >> Profits_500_0.1.txt
  python markow.py 5 250 >> Profits_5_250.txt
  python markow.py 50 250 >> Profits_50_250.txt
  python markow.py 500 250 >> Profits_500_250.txt
  python markow.py 5 500 >> Profits_5_500.txt
  python markow.py 50 500 >> Profits_50_500.txt
  python markow.py 500 500 >> Profits_500_500.txt
done
