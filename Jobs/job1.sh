cd ~
cd AnimalsRecognition
echo "Working in " `pwd`
echo "Contents of configuration"
cat Conf/dataset.json
echo "Running Driver"
python run.py
echo "Driver Run Complete"
