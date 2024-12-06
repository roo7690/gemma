import os
from argparse import ArgumentParser, ArgumentTypeError
import transformers
def init(model:str,auto_model:str='AutoModelForCausalLM'):
  os.tokenizer=transformers.AutoTokenizer.from_pretrained(model)
  os.model=getattr(transformers,auto_model).from_pretrained(model)

def chat():
  inputs=input()
  fn=handler.get(inputs,_chat(inputs))
  fn()
  chat()

handler={
  ":q":exit
}

def _chat(_input:str):
  def __chat():
    inputs=os.tokenizer(_input,return_tensors="pt")
    output=os.model(**inputs)
    print(os.tokenizer.decode(output.logits,skip_special_tokens=True))
  return __chat

if __name__=='__main__':
  parser=ArgumentParser()
  parser.add_argument('-M',type=str,help='Sélection du modèle à utiliser')
  parser.add_argument('-l',type=str,help='La classe utilise pour charger le modele')
  args=parser.parse_args()
  if args.M is None:
    raise ArgumentTypeError('Aucun modèle choisi')
  if args.l:
    init(args.M,args.l)
  else:
    init(args.M)
  print(f'Lancement de chat avec {os.model.config.model_type}')
  chat()