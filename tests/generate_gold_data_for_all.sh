echo "If it says 'unrecognized argument: --generate-data', don't worry. Not all scripts have this flag."
flist=$(ls test_*.py)
for f in $flist; do
    python $f --generate-data
    python $f --generate-gold
done