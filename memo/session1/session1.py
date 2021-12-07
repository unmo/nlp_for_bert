"""
# Session1のメモ及びコード

* 自然言語処理
  * 検索エンジン
  * 機械翻訳
  * 予測変換
  * スパムフィルタ
  * 音声アシスタント
  * 小説執筆
  * 対話システム
  * etc

* 形態素解析
  * 文章を単語に分割する技術
* 単語の分散表現
  * 文書内での関係性を踏まえて、単語をベクトル化する技術
* 再帰型NN(RNN)
  * 時系列を扱うのが得意なNN
* Seq2Seq
  * RNNベースの文章生成可能なモデル
* etc

* one-hot表現
  * １つだけ1で、その他は0のベクトルで単語を表現する方法
  
* 分散表現
  * 単語間の関連性や類似度に基づくベクトルで、単語を表現する
  * word2vecなどを使う。これを使えば、足し算引き算が可能なベクトルを作れる

* word2vec
  * 分散表現を作成するための技術
  * CBOW(continuous bag-of-words) or skip-gramが用いられる

* CBOW
  * 前後の単語から対象の単語を予測するNN
  * skip-gramより軽量

* skip-gram
  * ある単語から、前後の単語を予測するNN

* transformer
  * Attention層のみで構築される
  * 並列化が容易であり、訓練時間を大きく削減できる
  * Attention Is All You Need

* bert
  * googleのモデル
  * ベースはtransformer
  * finetuning可能→汎用性が高い
  * Transformerが、文脈から文脈を双方向に学習
  * MaskedLanguageModel及びNextSentencePredictionによる双方向学習

* MaskedLanguageModel
  * 文章から特定の単語を15%ランダムに選び、[MASK]トーケンに置き換える
  * my dog is hairy → my dog is [MASK]
  * [MASK]の単語を、前後の文脈から予測するように訓練する
  
* NextSentencePrediction
  * 2つの文章に関係があるかどうかを判定する
  * 後ろの文章を50%の確立で無関係な文章に置き換える
  * 後ろの文章が意味的に適切であればIsNext、そうでなければNotNextの判定

* SQuAD
  * スタンフォードが一般公開している、言語処理の精度を測るベンチマークデータセット
  
* GLUE
  * 自然言語処理のための9種類の学習データを含むデータセット
"""