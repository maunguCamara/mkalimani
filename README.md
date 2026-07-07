Existing models: 
 	Swahili(Helsinki,  NLLB, Masakhane)
 	Kikuyu(gatermark, )
 	Kamba(NLLB, Masakhane)
 	Dholuo(Helsinki-NLP, NLLB)
 	Somali(NLLB, MADLAD-400)
 	Oromo(Helsinki, MADLAD-400)

 Dataset but no model
 	Luhya, Kalenjin, Dawida(Taita), Maasai

 No coverage
  Gusii(ekegusii)
  Meru
  Giriama/Mijikenda
  Turkana
  Borana

Kikuyu model is finetuned from Googles, translategemma-4b-it base using rsLoRA ( rank 256, alpha 256), trained on 30, 430 English-kikuyu sentence pairs with Unsloth+ TRL on an H!00, achieving BLeU 21.93 and chrF ++ 42.87.
Kencorpus Luhya datset is Luhya-Kiswahili no English model
Bible corpus CHimoto and Basset for English-Luhya pairs from the New testament
Low word-pairs numbers thus chances of BLEU are high unless aggressive augmenting

Data collection per dialect
Start with Lunyole and Lukisa(can speak both, easy dta collection)
TranslateGemma saw very little Luhya text in pretraingin, more epochs or higher LoRA rank than Kikuyu.  sLoRA scaling alpha/sqrt(r) is designed for this - high rank without instability 


Data preparation -
Handle various data sources, KenTrans, Bible corpus, CSV or JSONL from local contributions, NLLB pivot to tranlste Swahili side of KenTrans to English via facebook/nllb-200-disltilled-600M. Score each pair using model's own beam search output. Those with low score do not enter training data. Can raise or lower scores based on data


Augment_data
Length-ration filter to remoovegarbage pairs where (en_len/luh_len is off), length bucket analysis with warnings, and optional back translation paraphrsing via Helinski-NLP en <-> fr models

Train
target_lang_code = 'luy'
num_train_epochs =5 
early stopping with patience = 3

Modal Train 
Drop-in modal H100 launcher, same infra as Kikuyu model

Evaluate 
BLEU + chrF++ on the held-out eval split, with same sample output printed 

Upload to hub
Push dataset to Huggingface and generate a model card with scores filled.

For slef-contibuted pairs, focus on speech domains under represented in the Bible corpus, like market speech, family vocabulary, food, numbers.
# 1. Get data from Harvard Dataverse + place Bible files
# 2. Add your pairs to data/contributed/

python scripts/01_prepare_data.py --skip-pivot   # fast first run
python scripts/02_augment_data.py                # check length stats
modal run scripts/04_modal_train.py --smoke-test  # sanity check
modal run scripts/04_modal_train.py --epochs 5    # real run
python scripts/05_evaluate.py --model-dir training/final_lora

