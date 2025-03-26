import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='web_scraping.log'
)

def scrape_all_page_text(url):
    try:
        logging.info(f"Iniciando scraping da URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text_elements = {
            'paragraphs': [],
            'headers': [],
            'links': [],
            'lists': [],
            'tables': [],
            'other_text': []
        }
        
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                text_elements['paragraphs'].append(text)
        
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                text = header.get_text(strip=True)
                if text:
                    text_elements['headers'].append(f'H{i}: {text}')
        
        for a in soup.find_all('a'):
            text = a.get_text(strip=True)
            if text:
                href = a.get('href', '')
                text_elements['links'].append(f"{text} [Link: {href}]")
        
        for ul in soup.find_all(['ul', 'ol']):
            list_items = [li.get_text(strip=True) for li in ul.find_all('li')]
            text_elements['lists'].append("\n".join(f"- {item}" for item in list_items if item))
        
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                cols = [col.get_text(strip=True) for col in row.find_all(['th', 'td'])]
                table_data.append(" | ".join(cols))
            text_elements['tables'].append("\n".join(table_data))
        
        for element in soup.find_all(['div', 'span', 'section', 'article']):
            if not element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'ul', 'ol', 'li', 'table']):
                text = element.get_text(strip=True)
                if text and len(text.split()) > 3:  
                    text_elements['other_text'].append(text)
        
        logging.info("Scraping concluído com sucesso")
        return text_elements
    
    except Exception as e:
        logging.error(f"Erro durante o scraping: {str(e)}")
        return None

def export_to_excel(data, filename):
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Textos da Página"
        
        # Cabeçalhos
        headers = [
            "Tipo de Elemento",
            "Conteúdo",
            "Quantidade de Caracteres",
            "Quantidade de Palavras"
        ]
        ws.append(headers)
        
        # Adicionar dados
        for element_type, contents in data.items():
            for content in contents:
                char_count = len(content)
                word_count = len(content.split())
                
                row = [
                    element_type.capitalize(),
                    content,
                    char_count,
                    word_count
                ]
                ws.append(row)
        
        for column in ['A', 'B', 'C', 'D']:
            ws.column_dimensions[column].width = 30
        
        wb.save(filename)
        logging.info(f"Dados exportados para {filename}")
        return True
    
    except Exception as e:
        logging.error(f"Erro ao exportar para Excel: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Web Scraping Avançado ===")
    url = input("Digite a URL completa da página (incluindo http:// ou https://): ").strip()
    
    page_texts = scrape_all_page_text(url)
    
    if page_texts:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"textos_pagina_{timestamp}.xlsx"
        
        if export_to_excel(page_texts, excel_filename):
            print(f"\n✅ Dados extraídos e salvos com sucesso no arquivo: {excel_filename}")
            print("\nResumo da extração:")
            for key, value in page_texts.items():
                print(f"{key.capitalize()}: {len(value)} itens")
        else:
            print("\n❌ Ocorreu um erro ao exportar para Excel. Verifique o arquivo de log.")
    else:
        print("\n❌ Falha ao extrair dados da página. Verifique o arquivo de log.")
