# QA System

## Question Similarity Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```

### S-Bert Train


```python
#train model
%%time
%run -i /home/yqinamz/work/QA_BOT/workspace/src/CS-QASystem-Torch/src/cs_qa_system_torch/question_similarity/train_QSM_Sbert.py \
--pretrain_model_path '/home/yqinamz/output/quora_sts-bert-base-nli-mean-tokens-2020-08-17_20-53-14/' \
--model_name '/home/yqinamz/output/Sbert_test' \
--train_data_path '/home/yqinamz/work/QA_BOT/data/QSM_2nd_Launch/Train_model/train_ramdon_1vs4_ann_V3_qa210' \
--train_batch_size 64 \
--num_train_epochs 1 \
--gpu 2,3,4,5 
```

### Inference on Question-Pairs


```python
%%time
%run -i /home/yqinamz/work/QA_BOT/workspace/src/CS-QASystem-Torch/src/cs_qa_system_torch/question_similarity/QSM_inference.py \
--saved_model_path '/home/yqinamz/output/quora_sts-bert-base-nli-mean-tokens-2020-08-17_20-53-14/' \
--test_data_path '/home/yqinamz/QA_Bot/QA_EXP/EXP_1002/Mix_2USE_8R_1009/test_sample.tsv'
```

### Real-Time Prediction for QSM


```python
%%time
%run -i /home/yqinamz/work/QA_BOT/workspace/src/CS-QASystem-Torch/src/cs_qa_system_torch/question_similarity/QSM_Real_time_inference.py \
--model_name '/home/yqinamz/output/Qa210_ann_v3-sbert-2020-12-07_20-12-07/' \
--qa_bank_json '/home/yqinamz/work/QA_BOT/data/QSM_2nd_Launch/Final_json/qa_bank_214_1208.json' \
--test_data_path '/home/yqinamz/work/QA_BOT/data/QSM_1st_Launch/Sbert_Result/traffic_3000_matched_q.txt'
```

### Real Time Inference (QSM)


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from QSM_Real_time_inference import ScoringService
```


```python
%%time
data = '{"query":"close account"}'
out = ScoringService.predict(data)
print(out)
```


## Information Retrieval

### Training IR Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```


```python
## Here model_name_or_path could be any mapping that AutoModel supports if architecture is ['bi', 'cross'] 
## and model_name_or_path ould be any mapping that AutoModelForClassification if architecture is 'cross-default'
## if you want to add new model you need add new instance of in ir_model to get the right output
## NOTE: model_name_or_path COULD BE A LOCAL PATH OR CHECKPOINT DIRECTORY WHERE THE MODEL WILL START FINETINING FROM THAT POINT
## NOTE: model_type can be anythinf for now. It is not being used.

%%time
%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/information_retrieval/ir_train.py \
        --train_data_path '/data/QAData/InformationRetrievalData/amazon/train.tsv' \
        --test_data_path '/data/QAData/InformationRetrievalData/amazon/test.tsv' \
        --model_name_or_path 'bert-base-uncased' \
        --architecture 'cross' \
        --do_lower_case \
        --train_batch_size 64 \
        --test_batch_size 128 \
        --num_train_epochs 2 \
        --gpu 2,3,4,5 \
        --save_steps 1000 \
        --print_freq 500 \
        --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase/'
    
```

### FineTuning IR Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```


```python
%%time
%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/information_retrieval/ir_train.py \
        --train_data_path '/data/QAData/InformationRetrievalData/amazon/finetune_rank_train.tsv' \
        --test_data_path '/data/QAData/InformationRetrievalData/amazon/finetune_rank_test.tsv' \
        --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase/' \
        --architecture 'cross' \
        --do_lower_case \
        --train_batch_size 64 \
        --test_batch_size 128 \
        --num_train_epochs 2 \
        --gpu 2,3,4,5 \
        --save_steps 1000 \
        --print_freq 500 \
        --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_finetuned/'
```

### Bulk Inference IR Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```


```python
%%time
%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/information_retrieval/ir_inference_mlmodel.py \
    --test_data_path '/data/QAData/InformationRetrievalData/amazon/finetune_rank_test.tsv' \
    --test_batch_size 128 \
    --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase/' \
    --architecture 'cross' \
    --do_lower_case \
    --prediction_file 'prediction.csv' \
    --gpu 2,3 \
    --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_inference/'
```

### Real Time Inference IR Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/information_retrieval/')
from ir_inference_mlmodel import ScorePrediction
import pandas as pd
pd.set_option('display.max_colwidth', None)
```


```python
%%time
query = 'how to cancel my prime membership'
passage = 'About Prime Gift Membership Cancellations: If you need to cancel a Prime gift membership that hasn\'t already been sent or has been sent but not redeemed, please contact us. Once a Prime gift membership has been redeemed by the recipient, it can\'t be canceled. The scheduled delivery date also can\'t be changed once your purchase has completed. If you need to update a delivery date, contact us so we can cancel the existing gift. You can then place a new order.'
out = ScorePrediction.get_document_score(query,passage)
print(out)
```

### Bulk Inference (BM25 + IR) Model 


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```


```python
%%time
%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/information_retrieval/ir_inference_combined.py \
    --passage_collection_path '/data/QAData/InformationRetrievalData/amazon/collection.json' \
    --qrels_path '/data/QAData/InformationRetrievalData/amazon/qrels.json' \
    --ir_model_name 'BM25Okapi' \
    --word_tokenizer_name 'simple_word_tokenizer' \
    --index_top_n 50 \
    --model_name_or_path 'output_torch/cross/ir_artifacts/bertbase_finetuned/' \
    --architecture 'cross' \
    --do_lower_case \
    --test_batch_size 512 \
    --gpu 2,3 \
    --prediction_file 'predictions.csv' \
    --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_inference/'
```

### Real Time Inference (BM25 + IR) Model


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from information_retrieval.ir_inference_combined import ScorePrediction
import pandas as pd
pd.set_option('display.max_colwidth', None)
```


```python
%%time
query = 'yes'
out = ScorePrediction.get_documents(query)
```

    CPU times: user 54 s, sys: 25 s, total: 1min 18s
    Wall time: 2.52 s



```python
out

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>qid</th>
      <th>pid</th>
      <th>query</th>
      <th>passage</th>
      <th>ir_score</th>
      <th>ml_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>46</td>
      <td>yes</td>
      <td>Unlimited Cloud Storage for Photos on Fire Tablets: Last updated: September 17, 2014. Fire tablet owners get free, unlimited photo storage, in full resolution, for all of their photos taken with their Fire tablet. Do I have to register my Fire tablet to get free unlimited photo storage from Amazon Drive? Yes. You will have to register your Fire tablet with your Amazon account to enjoy this benefit. Will my original photos be uploaded? Yes. Your photos will be uploaded and saved in your Amazon Drive account as-is. Can I upload just photos? No. When you turn on Auto-Save, your Fire tablet will upload both photos and videos. What happens to my video uploads when I go over my Amazon Drive storage limit? When you go over your storage limit, your Fire tablet will not upload your videos. Photos taken with your Fire tablet will continue to be uploaded and will not count against your storage limit. You can buy more storage or delete existing content on Amazon Drive to make room for videos to upload. I want to give my Fire tablet to my daughter. Will she get unlimited photo storage? Yes, your daughter will get the unlimited storage benefit when she registers the Fire tablet with her Amazon account. Will I lose the photos and videos that I have previously uploaded when I give away my Fire tablet? No. You can continue to view and download your photos and videos uploaded from your Fire tablet from Amazon Drive. When you buy a new Fire tablet, the uploaded photos and videos from your previous Fire tablet will be available on your new Fire tablet when you register your device with your Amazon account. I am an existing Amazon Drive customer; will I get unlimited photo storage for my new Fire tablet? Yes. When you register your new Fire tablet, you will automatically enjoy unlimited storage for all the photos you take with your new Fire tablet. Your Amazon Drive account will reflect this new benefit. Additional Terms &amp; Conditions. The unlimited photo storage benefit (the "Benefit") is not transferable, exchangeable or redeemable for cash and only applies to photos taken and uploaded from a Fire tablet registered to your Amazon account. Your use of the Benefit is subject to the [Cloud Drive Terms of Use](http://www.amazon.com/cd/tou). Void where prohibited by law. We reserve the right to terminate or modify this offer at any time.</td>
      <td>9.187283</td>
      <td>0.041589</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>102</td>
      <td>yes</td>
      <td>Redeem a Gift Card: Once applied to your Amazon account, the entire amount will be added to your gift card balance. Your gift card balance will be applied automatically to eligible orders during the checkout process and when using 1-Click. If you don't want to use your gift card balance on your order, you can unselect it as a payment method in checkout. Need to redeem a gift card? [Redeem a Gift Card](https://www.amazon.com/gp/css/gc/payment/view-gc-balance?). Locate the claim code. Note: For plastic gift cards, you may need to scratch off the coating on the back of the card to reveal the claim code. The claim code isn't the 16-digit card number. Go to [Your Account](https://www.amazon.com/gp/css/your-account-access). Click Apply a Gift Card to Your Account. Enter your claim code and click Apply to Your Balance. Note: You can also enter your claim code during checkout. You won't be able to redeem your gift card using the Amazon.com 1-Click service unless you redeem the gift card funds to Your Account first. Note: If your order total is more than your gift card funds, the extra can be paid for by credit card.</td>
      <td>0.000000</td>
      <td>0.112079</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>104</td>
      <td>yes</td>
      <td>Resolve a Declined Payment: To protect your security and privacy, your bank can't provide Amazon with information about why your payment was declined. Contact your bank directly to solve these payment issues. To retry a declined payment, go to [Your Orders](www.amazon.com/gp/your-account/order-history). To determine why your payment was declined, consider the following, and, if necessary, contact your bank for more information: Have you exceeded your credit limit? Did you enter your credit card number, credit card expiration date, billing address, and phone number correctly in [Your Account](www.amazon.com/youraccount)? Is your purchase outside of your normal spending range? Some banks will block transactions due to security concerns. Does your issuing bank have special policies regarding electronic or internet purchases? To retry a declined payment: Go to [Your Orders](www.amazon.com/gp/your-account/order-history). Do one of the following: Try again with a different payment method, as follows: Select Change Payment Method next to the order you want to modify. Select another payment method from your account or submit a new card number and select Confirm. Select Retry Payment Method next to the order. Retry with your current payment method by selecting Retry Payment Method next to the order.</td>
      <td>0.000000</td>
      <td>0.120437</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>105</td>
      <td>yes</td>
      <td>Track Your Package on a Mobile Device: You can track your orders through Your Account on your mobile device, even if the order wasn't originally placed through that device. To track your order: From the Amazon App for your iPhone/iPad, tap More. From your Android phone, tap the three horizontal bars in the upper, left of the home screen. Tap Your Orders. Find the order you want to view and tap View, change, or track order if using a mobile application, or Track Item if using a mobile browser. Note: If the order you wish to view is not displayed, change the Orders placed in the last 6 months option at the top of the Order history screen.</td>
      <td>0.000000</td>
      <td>0.027721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>96</td>
      <td>yes</td>
      <td>Return a Kindle Book Order: Cancel an accidental book order within seven days.  Approved refunds are credited to the original payment source within three to five days. Need to cancel a Kindle book order? [Start Your Kindle Book Return](www.amazon.com/digitalorders). Go to [https://www.amazon.com/digitalorders](https://www.amazon.com/digitalorders) and sign in with the same Amazon account information you used to purchase your content. From the Digital Orders tab, select the Return for Refund button next to the title you want to return. In the pop-up window, choose the reason for return, then select Return for Refund. Tip: Prevent accidental purchases by setting parental controls.</td>
      <td>0.000000</td>
      <td>0.019409</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>95</td>
      <td>yes</td>
      <td>Cancel Amazon Music HD Subscription: : You can cancel your Amazon Music HD subscription at any time from Your Amazon Music Settings. Go to [Your Amazon Music Settings](https://www.amazon.com/music/settings). Select Cancel Subscription. Confirm the cancellation. If you wish to continue to subscribe to Amazon Music Unlimited without HD: Select Remove HD from my subscription. Confirm the cancellation. Note: Your subscription end date displays on the confirmation screen. Though you'll no longer be charged, you can continue to access titles in Amazon Music HD up until this date. After this date, any Amazon Music HD titles you've added to My Music will be greyed out, and will need to be re-downloaded in Standard quality for any offline playback.</td>
      <td>0.000000</td>
      <td>0.016682</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>94</td>
      <td>yes</td>
      <td>Find a Missing Package That Shows As Delivered: What to do if your package shows as delivered but you can't find it. If your tracking information shows that your package was delivered, but you can't find it: Within 48 hours of expected delivery. Verify the shipping address. Look for a notice of attempted delivery. Look around the delivery location for your package. See if someone else accepted the delivery, unless you have health or safety concerns about doing so. Some packages travel through multiple carriers; check your mailbox or wherever else you receive mail. Wait 48 hours - in rare cases packages may say delivered up to 48 hours prior to arrival</td>
      <td>0.000000</td>
      <td>0.054730</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>93</td>
      <td>yes</td>
      <td>Return Your Rental: You can return your rental through Your Account within the Initial Rental Refund Period to receive a full refund. If you return your rental after the Initial Rental Refund Period, you won’t be eligible for a refund of the rental fee. Need to return a Rental after 30 days? Go to [Manage Your Rentals](https://www.amazon.com/gp/rental/your-account?ie=UTF8&amp;ref_=ya_rentals&amp;). To return a rental after the initial rental period: Go to [Manage Your Rentals](https://www.amazon.com/gp/rental/your-account). Select the rental item you wish to return and then select Return rental to print the pre-paid return shipping label. Print out the packaging slip and return shipping label. Package up the textbook you wish to return, including the packaging slip. Note: To avoid incorrect fees, only rentals listed on the same packing slip should be packaged and returned together. Apply the return shipping label and take the shipment to the carrier listed on your return label. Return shipping is free when you use the shipping label provided. Note: To return a rental within the Initial Rental Refund Period, go to [Manage Your Rentals](https://www.amazon.com/gp/rental/your-account?ie=UTF8&amp;ref_=ya_rentals&amp;). If the item is in the same condition as when you received it, you’ll receive a full refund. If the item is damaged during the rental period, additional damage fees may apply. To avoid automatic extension fees, please drop your items off with the carrier on or before the due date. For more information on Rental Return periods, visit [Rentals Terms and Conditions](www.amazon.com/gp/help/customer/display.html?nodeId=201983840).</td>
      <td>0.000000</td>
      <td>0.090762</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>92</td>
      <td>yes</td>
      <td>Exchange an Item: You can exchange items that qualify through Your Orders. If the item doesn't have an exchange option or you received the item as a gift, you'll need to return the original item and place a new order. To exchange an item: Go to [Your Orders](www.amazon.com/gp/css/order-history) and select Return or replace items beside the item you want to exchange. Tip: If you don't see the order you're looking for, select the date on which you placed the order from the drop-down menu in the upper part of the page. Select the item you want to exchange, and select a reason from the Reason for return menu.  An exchange order will be created with the same shipping speed that was used on your original item. Use the return label provided to send your original item back. Return the original item within 30 days to avoid being charged for both the original and exchange items. This policy refers to color and size exchanges only. Amazon doesn't price match. If the item you're exchanging costs less than your original purchase, we'll refund you the difference. If the new item costs more, we'll charge you for the difference in price. For details on what types of items can be exchanged, see [Exchanges and Replacements](www.amazon.com/gp/help/customer/display.html?nodeId=GYG97JCDCUNGURQJ).</td>
      <td>0.000000</td>
      <td>0.273875</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>91</td>
      <td>yes</td>
      <td>Managing Your Amazon Prints Orders: You can cancel your order within one hour of placement or reorder a previous project. For prompt delivery, we begin creating your order one hour after you place it. After you place your order, you will see a timer set to one hour. Once the timer expires, your order starts processing and you're unable to cancel. If you would like to edit your order, please cancel it within this hour and make the desired changes by going to My Projects.Cancel=To cancel an order within an hour of submitting it: Go to Your Orders on Amazon.com. Click Cancel items. Confirm by selecting Cancel selected items in this order. Note: If you used an Amazon Prints promotional credit to pay for the order, the credit amount will return to your account when you cancel the order, and visible on the Credits tab..Track=To track your order: Go to Your Orders on Amazon.com. Click on Tracking Number. If you experience any issues contact us within 30 days of receiving the order and we'll work with you to make it right. Note: Damage during order production is rare; in some cases, we may ask you to submit an image of the damage.</td>
      <td>0.000000</td>
      <td>0.008931</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = [ScorePrediction.transform(query,i[3]) for i in out]
type(x[0])
```


```python
%%time
import torch
with torch.no_grad():
    ScorePrediction.ml_model(*out)
```

## Answer Extraction

### Train Answer Extraction Model


```python
%%time
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/answer_extraction/run_squad_final.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --do_train \
    --do_eval \
    --data_dir '/data/QAData/AnswerExtractionData/amazon' \
    --train_file 'train_answerextraction_100.tsv' \
    --predict_file 'test_answerextraction_100.tsv' \
    --squad_data_dir '/data/QAData/AnswerExtractionData/squad' \
    --squad_train_file 'train-v2.0.json' \
    --squad_predict_file 'dev-v2.0.json' \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --add_neg \
    --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final'
```

### Fine Tuning Answer Extraction with New Amazon Data


```python
import pandas as pd
realtime_new_annotations = [
    {
    "qid":20000,
    "query":"cancel my music subscription",
    "answer":"If you subscribed to Amazon Music Unlimited through iTunes visit the [Apple website](https://support.apple.com/HT202039) to cancel your subscription. If you subscribed through a third-party, such as a mobile service provider, contact them for further assistance. Go to [Your Amazon Music Settings](https://www.amazon.com/music/settings). Go to the Amazon Music Unlimited section. Select the Cancel option in your Subscription Renewal details. Confirm the cancellation",
    "passage":"Cancel Amazon Music Unlimited Subscription: Cancel your subscription online. If you subscribed to Amazon Music Unlimited through iTunes visit the [Apple website](https://support.apple.com/HT202039) to cancel your subscription. If you subscribed through a third-party, such as a mobile service provider, contact them for further assistance. Go to [Your Amazon Music Settings](https://www.amazon.com/music/settings). Go to the Amazon Music Unlimited section. Select the Cancel option in your Subscription Renewal details. Confirm the cancellation. Note: You can continue to access Amazon Music Unlimited until the end date. After this date, any Amazon Music Unlimited titles you've added to Library will be grayed out, with playback options removed.Cancelling an Amazon Music Unlimited subscription also ends membership to Amazon Music HD. Video: Cancel Amazon Music Unlimited. For more help, try our [Amazon Music forum](https://www.amazon.com/gp/redirect.html/ref=hp_cr_music_forum?location=https://www.amazonforum.com/s/amazon-music&token=728A8569DD99E77B59F36EE4739090DC4E1BA6F8&source=standards). [](https://www.amazon.com/gp/redirect.html/ref=hp_cr_music_forum?location=https://www.amazonforum.com/s/amazon-music&token=728A8569DD99E77B59F36EE4739090DC4E1BA6F8&source=standards)"
    }
]

processed_new_annotations = '/data/QAData/AnswerExtractionData/amazon/finetune_answer_train.tsv'
#amazon_old_annotations = '/data/QAData/AnswerExtractionData/amazon/train_answerextraction_100.tsv'

df_realtime_new_annotations = pd.DataFrame(realtime_new_annotations)
df_processed_new_annotations = pd.read_csv(processed_new_annotations,sep='\t')
#df_amazon_old_annotations = pd.read_csv(amazon_old_annotations,sep='\t')

#df_final = pd.concat([df_realtime_new_annotations, df_processed_new_annotations, df_amazon_old_annotations])
df_final = pd.concat([df_realtime_new_annotations, df_processed_new_annotations])
df_final = df_final.sample(frac=1).reset_index(drop=True)
df_final.to_csv('/data/QAData/AnswerExtractionData/amazon/train_finetune_final.tsv',sep='\t',index=None)
```


```python
%%time
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/answer_extraction/run_squad_final.py \
    --model_type bert \
    --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final' \
    --do_lower_case \
    --do_train \
    --do_eval \
    --data_dir '/data/QAData/AnswerExtractionData/amazon' \
    --train_file 'train_finetune_final.tsv' \
    --predict_file 'test_answerextraction_100.tsv' \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --add_neg \
    --overwrite_output_dir \
    --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final_finetune'
```

### Inference of Answer Extraction Model


```python
%%time
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/answer_extraction/run_squad_final.py \
  --model_type bert \
  --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final_finetune' \
  --do_lower_case \
  --do_eval \
  --data_dir '/data/QAData/AnswerExtractionData/amazon' \
  --predict_file 'finetune_answer_test.tsv' \
  --per_gpu_eval_batch_size 128 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --add_neg \
  --overwrite_cache \
  --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_inference'

```

### Testing of Answer Extraction Model


```python
%%time
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')

%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/answer_extraction/run_squad_final.py \
  --model_type bert \
  --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final' \
  --do_lower_case \
  --do_test \
  --data_dir '/data/QAData/AnswerExtractionData/amazon' \
  --test_file 'test_answerextraction_100.tsv' \
  --per_gpu_test_batch_size 128 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --add_neg \
  --overwrite_cache \
  --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_test'
```

### Real Time Inference Answer Extraction


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

output_dir = '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output_final_finetune'
do_lower_case = True
model = AutoModelForQuestionAnswering.from_pretrained(output_dir)  # , force_download=True)
tokenizer = AutoTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
device = "cpu"
model.to(device)

```


```python
import torch
from transformers.data.processors.squad import SquadExample, squad_convert_example_to_features, squad_convert_example_to_features_init
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.data.processors.squad import SquadResult

def to_list(tensor):
    return tensor.detach().cpu().tolist()

#question = 'who is rajesh friend'
#context = 'Manu likes swimming. His age is 23 years. Manu has friend Rajesh, whose age is 52 years.'

question= "unknown charge on my account"
context = "Unknown Charges: There are several reasons why you might not recognize a charge. If you want to review your complete order history, go to [Your Orders](www.amazon.com/your-orders). The following are common scenarios for unknown charges: An Amazon Prime yearly subscription was renewed. For more information, go to [Manage Your Prime Membership](www.amazon.com/gp/subs/primeclub/account/homepage.html?ie=UTF8&ref_=nav_youraccount_prime). A bank has placed an authorization hold for recently canceled or changed orders. When you place an order, Amazon contacts the issuing bank to confirm the validity of the payment method. Your bank reserves the funds until the transaction processes or the authorization expires, but this isn't an actual charge. If you cancelled your order, the authorization will be removed from your account according to the policies of your bank. To remove an authorization, contact your bank to clarify how long they hold authorizations for online orders. An order was placed by a family member, friend, or co-worker with access to your card number. Additional cards are associated with the credit or debit account. A back-ordered or pre-ordered item shipped. A gift order shipped. An order placed outside Amazon.com using Amazon Pay. Amazon Pay orders begin with ‘P01’ and are followed by 14 digits. Check your Amazon Pay Account for your order history. For further assistance with any Amazon Pay transactions, see the Amazon Pay Help pages. An order was split into multiple shipments or sent to multiple shipping addresses. Note: This appears on your statement as separate charges. If the charge isn't explained by any of these situations, contact us by phone and have the following information available: Date of the charge. Amount of the charge. Your name, email address, and phone number. Charge ID, a unique 9-digit alphanumeric code that Customer Service can use to locate your charge, e.g., Amazon.com*A123B4CD5, Prime Now*B123B4CD5, AMZN Mktp US*C123B4CD5. Note: This ID only appears when the card is charged. It is not present on Pending charges (Authorizations). Use your order history to review your orders and shipments to compare the charges with the charges on your bank statement..Cancel Subscribe with Amazon Subscriptions: You can cancel your Subscribe with Amazon subscriptions at any time from Your Memberships and Subscriptions in Your Account. To cancel a subscription you purchased using Subscribe with Amazon: Go to [Your Memberships and Subscriptions](https://www.amazon.com/yourmembershipsandsubscriptions). Select Manage Subscription next to the subscription you'd like to cancel. Click the link under Advance Controls to be directed to the main subscription page. From this page, you can end your subscription. About Cancellations and Refunds for Digital Subscriptions. Once you cancel, the renewal date in your subscription details becomes the end date. You'll no longer be charged for the subscription, but can continue to access it until this date. You can cancel a subscription at any time, with no early termination fees. For subscriptions with a renewal period longer than one month, new subscribers are eligible for a full refund if they cancel within 7 days of purchase. Access to the subscription is removed as soon as a refund is issued. Outside of the refund window, or for subscriptions with a billing period one month or less in length, cancellation turns off auto-renew for the subscription so you won't be billed again. Once you cancel you will still have access to the subscription through the end of your current paid billing period. For subscriptions with a billing period of 6 months or longer, or when otherwise required by law, we'll notify you when the subscription is set to renew. You can then change your billing information or cancel your subscription before you're charged again. Note: To learn more about auto-renewals, go to [About Auto-Renewal for Subscribe with Amazon Subscriptions (Digital Subscriptions Only)](https://www.amazon.com/gp/help/customer/display.html/?nodeId=202013370). About Cancellations for Physical Subscriptions. Once you cancel, no further orders will be placed. You will not be charged until the order ships. Cancelling a subscription does not cancel any pending unshipped order. To cancel pending unshipped orders, you can go to Your Orders and request cancellation"
max_seq_length = 384
doc_stride = 128
max_query_length = 64
model_type = 'bert'
n_best_size = 1
max_answer_length = 150
do_lower_case = True
example = SquadExample(
            qas_id=0,
            question_text=question,
            context_text=context,
            answer_text=None,
            start_position_character=None,
            title=None,
            answers=None,
        )
squad_convert_example_to_features_init(tokenizer)
features = squad_convert_example_to_features(example, max_seq_length, doc_stride, max_query_length, False)

new_features = []
unique_id = 1000000000
for feature in features:
    feature.example_index = 0
    feature.unique_id = unique_id
    new_features.append(feature)
    unique_id += 1
features = new_features
del new_features    

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
dataset = TensorDataset( all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask)
sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

all_results = []
for batch in dataloader:
    model.eval()
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
             "token_type_ids": batch[2],
        }

        if model_type in ["xlm", "roberta", "distilbert", "camembert", "bart"]:
            del inputs["token_type_ids"]

        feature_indices = batch[3]

        # XLNet and XLM use more arguments for their predictions
        if model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
            # for lang_id-sensitive xlm models
            if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                inputs.update(
                    {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                )
        outputs = model(**inputs)


    for i, feature_index in enumerate(feature_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)

        output = [to_list(output[i]) for output in outputs]

        # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
        # models only use two.
        if len(output) >= 5:
            start_logits = output[0]
            start_top_index = output[1]
            end_logits = output[2]
            end_top_index = output[3]
            cls_logits = output[4]

            result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

        else:
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
        all_results.append(result)

predictions_realtime = compute_predictions_logits(
        [example],
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        None,
        None,
        None,
        True,
        True,
        0.0,
        tokenizer,
        )
print(context,'\n\n')
print(predictions_realtime[0])
```

## (Information Retrieval + Answer Extraction)

### Bulk Inference (IR + AE)


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
```


```python
%%time
%run -i /home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/ir_and_ae/ir_ae_inference.py \
        --passage_collection_path '/data/QAData/InformationRetrievalData/amazon/goldlabel_collection.json' \
        --qrels_path '/data/QAData/InformationRetrievalData/amazon/goldlabel_qrels.json' \
        --ir_model_name 'BM25Okapi' \
        --word_tokenizer_name 'simple_word_tokenizer' \
        --index_top_n 50 \
        --ir_model_weight 0.46 \
        --model_name_or_path '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase/' \
        --architecture 'cross' \
        --do_lower_case \
        --ir_top_n 2 \
        --ae_params_dir '/home/srikamma/efs/work/QASystem/QAModel/output_answer_extraction/bert_output' \
        --version_2_with_negative \
        --max_seq_length 384 \
        --max_answer_length 100 \
        --doc_stride 128 \
        --test_batch_size 512 \
        --prediction_file 'prediction.csv' \
        --gpu 2,3 \
        --output_dir '/home/srikamma/efs/work/QASystem/QAModel/output_torch/cross/ir_artifacts/bertbase_inference_ir_ae/'


#         --passage_collection_path '/data/QAData/InformationRetrievalData/amazon/collection.json' \
#         --qrels_path '/data/QAData/InformationRetrievalData/amazon/qrels.json' \
    
```


```python
results
```

### Real Time Inference (IR+AE)


```python
import sys
sys.path.append('/home/srikamma/efs/workspace/CS-QASystem-Torch/src/CS-QASystem-Torch/src/cs_qa_system_torch/')
from ir_and_ae.ir_ae_inference import ScorePrediction
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/srikamma/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
%%time
query='unknown charge'
out = ScorePrediction.get_answer(query)
print(out)
```

    INSIDE THE INIT!!
    [{'type': 'ANSWER', 'value': 'The following are common scenarios for unknown charges: An Amazon Prime yearly subscription was renewed', 'relativeUrl': '', 'desktopRelativeUrl': '', 'buttons': []}, {'type': 'ANSWER', 'value': 'For more information, go to <a>Manage Your Prime Membership</a>. A bank has placed an authorization hold for recently canceled or changed orders', 'relativeUrl': 'www.amazon.com/gp/subs/primeclub/account/homepage.html?ie=UTF8&ref_=nav_youraccount_prime', 'desktopRelativeUrl': 'www.amazon.com/gp/subs/primeclub/account/homepage.html?ie=UTF8&ref_=nav_youraccount_prime', 'buttons': []}, {'type': 'ANSWER', 'value': 'When you place an order, Amazon contacts the issuing bank to confirm the validity of the payment method', 'relativeUrl': '', 'desktopRelativeUrl': '', 'buttons': []}, {'type': 'ANSWER', 'value': "Your bank reserves the funds until the transaction processes or the authorization expires, but this isn't an actual charge", 'relativeUrl': '', 'desktopRelativeUrl': '', 'buttons': []}]
    CPU times: user 37.4 s, sys: 16.7 s, total: 54.1 s
    Wall time: 11.9 s

