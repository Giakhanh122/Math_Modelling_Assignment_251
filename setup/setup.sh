#! create python virtual environnent
python3 -m venv test

#! activate virtual environment
source test/bin/activate

#! pip upgrade
pip install --upgrade pip

#! install all needed library
pip install -r requirements.txt


echo "Setup done !"
