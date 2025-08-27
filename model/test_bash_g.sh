python google_inv.py --dataset gossipcop
python dem_inv.py --shot 100 --dataset gossipcop
python determine.py --shot 100 --dataset gossipcop --lc_use 1
python dem_inv.py --shot 64 --dataset gossipcop
python determine.py --shot 64 --dataset gossipcop --lc_use 1
python dem_inv.py --shot 32 --dataset gossipcop
python determine.py --shot 32 --dataset gossipcop --lc_use 1
python dem_inv.py --shot 16 --dataset gossipcop
python determine.py --shot 16 --dataset gossipcop --lc_use 1
python dem_inv.py --shot 8 --dataset gossipcop
python determine.py --shot 8 --dataset gossipcop --lc_use 1