mkdir -p data/trustpilot && cd data/trustpilot

wget https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/denmark.auto-adjusted_gender.NUTS-regions.jsonl.zip
wget https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/germany.auto-adjusted_gender.NUTS-regions.jsonl.zip
wget https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.zip
wget https://bitbucket.org/lowlands/release/raw/fd60e8b4fbb12f0175e0f26153e289bbe2bfd71c/WWW2015/data/united_states.auto-adjusted_gender.geocoded.jsonl.zip

unzip denmark.auto-adjusted_gender.NUTS-regions.jsonl.zip
unzip germany.auto-adjusted_gender.NUTS-regions.jsonl.zip
unzip united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.zip
unzip united_states.auto-adjusted_gender.geocoded.jsonl.zip

rm denmark.auto-adjusted_gender.NUTS-regions.jsonl.zip
rm germany.auto-adjusted_gender.NUTS-regions.jsonl.zip
rm united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.zip
rm united_states.auto-adjusted_gender.geocoded.jsonl.zip
rm -r __MACOSX/


cd ../..
