from transformers import GPT2Tokenizer
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
import PyPDF2
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

def obter_cabecalhos_planilha_matriz():
    df_matriz = pd.read_excel(r"C:\Users\youu!\Downloads\matriz.xlsx")
    cabecalhos = df_matriz.columns.tolist()
    return cabecalhos

tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')

def preencher_dados_com_IA_por_coluna(cabecalho, dados_usuario):
    prompt = f"""
Você é um assistente que ajuda a extrair informações de documentos.

Com base nas seguintes informações fornecidas pelo usuário:

{dados_usuario}

Preencha o seguinte campo correspondente ao cabeçalho:

{cabecalho}

Retorne a informação correspondente, separada por tabulação (\\t).
Se não souber a informação, deixe em branco.
"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if input_ids.size(1) > 2048:
        input_ids = input_ids[:, :2048]  #Truncar para 2048 tokens
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True) 

    set_seed(42)

    resposta = generator(prompt, max_new_tokens=150, num_return_sequences=1)

    texto_resposta = resposta[0]['generated_text'].strip()
    return texto_resposta

def processar_arquivo_usuario(file_path):
    cabecalhos = obter_cabecalhos_planilha_matriz()
    
    if file_path.endswith('.xlsx'):
        df_usuario = pd.read_excel(file_path)
        dados_usuario = df_usuario.to_string(index=False)
    elif file_path.endswith('.csv'):
        df_usuario = pd.read_csv(file_path)
        dados_usuario = df_usuario.to_string(index=False)
    elif file_path.endswith('.pdf'):
        texto = ""
        with open(file_path, 'rb') as file:
            leitor = PyPDF2.PdfReader(file)
            for pagina in range(len(leitor.pages)):
                texto += leitor.pages[pagina].extract_text()
        dados_usuario = texto
    else:
        print("Formato de arquivo não suportado.")
        return

    respostas = []
    for cabecalho in cabecalhos:
        resposta_coluna = preencher_dados_com_IA_por_coluna(cabecalho, dados_usuario)
        respostas.append(resposta_coluna)

    gerar_txt(cabecalhos, respostas)

def gerar_txt(cabecalhos, respostas):
    with open('resultado.txt', 'w', encoding='utf-8') as file:
        file.write('\t'.join(cabecalhos) + '\n')
        file.write('\t'.join(respostas) + '\n')
    print("Arquivo 'resultado.txt' gerado com sucesso.")

def carregar_arquivo():
    file_path = filedialog.askopenfilename(filetypes=[("Planilhas", "*.xlsx"), ("CSV", "*.csv"), ("PDF", "*.pdf")])
    if file_path:
        processar_arquivo_usuario(file_path)

def main():
    root = ctk.CTk()
    root.title("Preenchimento Automático de Tabelas")
    root.geometry("400x200")

    label = ctk.CTkLabel(root, text="Carregar Arquivo:")
    label.pack(pady=10)

    upload_button = ctk.CTkButton(root, text="Upload", command=carregar_arquivo)
    upload_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
