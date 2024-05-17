for corp in 'webnlg' 'e2e'
do
  python3 prepare_corpus.py --aspect systematicity --sys atom --corp $corp
  python3 prepare_corpus.py --aspect systematicity --sys combination --corp $corp
done

for corp in 'webnlg' 'e2e'
do
  for pro_len in 3 4 5
  do
    python3 prepare_corpus.py --aspect productivity --pro invisible --pro_len $pro_len --corp $corp
    python3 prepare_corpus.py --aspect productivity --pro visible --pro_len $pro_len --corp $corp
  done
done

for corp in 'webnlg' 'e2e'
do
  python3 prepare_corpus.py --aspect order-invariance --ord original --corp $corp
  python3 prepare_corpus.py --aspect order-invariance --ord match --corp $corp
  python3 prepare_corpus.py --aspect order-invariance --ord_aspect main --corp $corp
  python3 prepare_corpus.py --aspect order-invariance --ord_aspect cwio --corp $corp
  python3 prepare_corpus.py --aspect order-invariance --ord_aspect performance --corp $corp
done

for corp in 'webnlg' 'e2e'
do
  python3 prepare_corpus.py --aspect rule-learnability --corp $corp
done
