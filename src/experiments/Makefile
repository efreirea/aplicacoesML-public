DATABASE = ../../../union_gch_rw/metabase_theilu.csv
METALEARNER = ../metalearner.py

onlyC1_CFS.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_2 -igCol sector  -fs cfs > onlyC1_CFS.txt
onlyC1_RFE.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_2 -igCol sector  -fs rfe > onlyC1_RFE.txt

onlyC2_CFS.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_1 -igCol sector  -fs cfs > onlyC2_CFS.txt
onlyC2_RFE.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_1 -igCol sector  -fs rfe > onlyC2_RFE.txt

onlyC0_CFS.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol sector  -fs cfs > onlyC0_CFS.txt
onlyC0_RFE.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol sector  -fs rfe > onlyC0_RFE.txt

onlySector_CFS.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol Community_0  -fs cfs > onlySector_CFS.txt
onlySector_RFE.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol Community_0  -fs rfe > onlySector_RFE.txt

allCols_CFS.txt:
	python3 $(METALEARNER) -f $(DATABASE)  -fs cfs > allCols_CFS.txt
allCols_RFE.txt:
	python3 $(METALEARNER) -f $(DATABASE)  -fs rfe > allCols_RFE.txt

allCols_nofs.txt:
	python3 $(METALEARNER) -f $(DATABASE)  -fs nofs > allCols_nofs.txt

only_single_comm: onlyC1_CFS.txt onlyC1_RFE.txt onlyC2_CFS.txt onlyC2_RFE.txt onlyC0_CFS.txt onlyC0_RFE.txt

onlyC1_nofs.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_2 -igCol sector  -fs nofs > onlyC1_nofs.txt
onlyC2_nofs.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_0 -igCol Community_1 -igCol sector  -fs nofs > onlyC2_nofs.txt
onlyC0_nofs.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol sector  -fs nofs > onlyC0_nofs.txt
onlySector_nofs.txt:
	python3 $(METALEARNER) -f $(DATABASE) -igCol Community_2 -igCol Community_1 -igCol Community_0  -fs nofs > onlySector_nofs.txt


nofs: allCols_nofs.txt onlyC1_nofs.txt onlyC2_nofs.txt onlyC0_nofs.txt onlySector_nofs.txt

run_exp: only_single_comm allCols_CFS.txt allCols_RFE.txt nofs onlySector_CFS.txt onlySector_RFE.txt
