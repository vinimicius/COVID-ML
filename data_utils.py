import zipfile
import os
import pandas as pd



def extract(data_dir='data'):
    """
    Localiza o ZIP no diretório especificado, extrai e padroniza o nome para data_raw.csv.
    
    Args:
        data_dir (str): Caminho para a pasta que contém o data.zip. 
                        O padrão é 'data', relativo à raiz do projeto.
    
    Returns:
        str: Caminho completo para o arquivo CSV extraído.
    """
    zip_path = os.path.join(data_dir, 'data.zip')
    raw_csv_path = os.path.join(data_dir, 'data_raw.csv')

    # 1. Verificação de Idempotência (Não repetir trabalho já feito)
    if os.path.exists(raw_csv_path):
        print(f"ℹ️ O arquivo {raw_csv_path} já existe. Pulando extração.")
        return raw_csv_path

    # 2. Validação de existência do ZIP
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"❌ Erro crítico: O arquivo {zip_path} não foi encontrado!")

    # 3. Processo de Extração
    print(f"Iniciando extração de {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Pegamos o nome do primeiro arquivo dentro do ZIP
            internal_files = zip_ref.namelist()
            if not internal_files:
                raise ValueError(f"❌ O arquivo {zip_path} está vazio!")
                
            zip_ref.extractall(data_dir)
            
            # Caminho onde o arquivo caiu originalmente
            original_extracted_path = os.path.join(data_dir, internal_files[0])
            
            # Renomeia para o nosso padrão data_raw.csv
            os.rename(original_extracted_path, raw_csv_path)
            print(f"✅ Sucesso! Dados extraídos em: {raw_csv_path}")
            
    except zipfile.BadZipFile:
        raise Exception(f"❌ O arquivo {zip_path} parece estar corrompido.")

    return raw_csv_path



def import_data(file_path):
    """
    Lê o arquivo CSV bruto do SUS e retorna um DataFrame.
    
    Args:
        file_path (str): Caminho completo para o arquivo CSV (ex: 'data/data_raw.csv').
        
    Returns:
        pd.DataFrame: O dataset carregado.
        
    Raises:
        FileNotFoundError: Se o arquivo não existir.
        RuntimeError: Se houver falha na leitura pelo Pandas.
    """
    # 1. Validação de existência (Falha rápida/Fail-fast)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Erro: O arquivo '{file_path}' não foi encontrado.")

    print(f"📖 Carregando dados de: {file_path}...")
    
    try:
        # 2. Leitura com os parâmetros otimizados para o OpenDataSUS
        df = pd.read_csv(
            file_path, 
            sep=';', 
            encoding='latin-1', 
            low_memory=False
        )
        
        # 3. Feedback visual imediato
        print(f"📊 Dataset carregado com sucesso: {df.shape[0]:,} linhas e {df.shape[1]} colunas.")
        return df
        
    except Exception as e:
        # Encapsulamos o erro técnico em uma mensagem legível para o pesquisador
        raise RuntimeError(f"❌ Falha crítica ao ler o CSV: {str(e)}")
    


def clean_data(df):
    """
    Realiza a limpeza técnica e filtragem de integridade do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame bruto carregado do CSV.
        
    Returns:
        pd.DataFrame: DataFrame limpo e reduzido.
    """
    if df is None:
        raise ValueError("❌ Erro: O DataFrame fornecido para limpeza é nulo.")

    initial_rows = len(df)
    
    # 1. Configurações de Limpeza (Centralizado para fácil manutenção)
    cols_to_remove = [
        'data_inicio_sintomas', 'codigo_ibge', 
        'diagnostico_covid19', 'nome_munic'
    ]
    
    disease_cols = [
        'asma', 'cardiopatia', 'diabetes', 'doenca_hematologica',
        'doenca_hepatica', 'doenca_neurologica', 'doenca_renal',
        'imunodepressao', 'obesidade', 'outros_fatores_de_risco',
        'pneumopatia', 'puerpera', 'sindrome_de_down'
    ]

    print("🧹 Iniciando limpeza dos dados...")

    # 2. Remoção de Colunas (Drop antecipado para liberar memória)
    df_cleaned = df.drop(columns=cols_to_remove).copy()
    print(f"   - {len(cols_to_remove)} colunas desnecessárias removidas.")

    # 3. Filtragem de Valores 'IGNORADO'
    # .all(axis=1) garante que a linha só fica se TODAS as colunas de doença forem válidas
    valid_mask = (df_cleaned[disease_cols] != 'IGNORADO').all(axis=1)
    df_cleaned = df_cleaned[valid_mask]
    print("   - Linhas com valores 'IGNORADO' descartadas.")

    # 4. Remoção de Valores Nulos (NaN) residuais
    df_cleaned = df_cleaned.dropna()
    print("   - Linhas com valores nulos (NaN) removidas.")

    # 5. Relatório de Eficiência
    final_rows = len(df_cleaned)
    retention_rate = (final_rows / initial_rows) * 100

    print(f"\n--- Relatório de Limpeza ---")
    print(f"✅ Dimensões finais: {final_rows:,} linhas e {df_cleaned.shape[1]} colunas.")
    print(f"📉 Retenção de dados: {retention_rate:.2f}% do volume original.")
    
    return df_cleaned



def pre_process(df):
    """
    Realiza a engenharia de recursos: binarização, encoding, escalonamento e tipagem.    
    Args:
        df (pd.DataFrame): DataFrame limpo pela função clean_data.
        
    Returns:
        pd.DataFrame: Dataset puramente numérico (inteiros).
    """
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


    # Opt-in para o novo comportamento do Pandas (silencia o FutureWarning)
    pd.set_option('future.no_silent_downcasting', True)
    
    if df is None:
        raise ValueError("❌ Erro: O DataFrame fornecido para pré-processamento é nulo.")
    
    # Trabalhamos em uma cópia para evitar efeitos colaterais no DF original
    df_proc = df.copy()

    # 1. Binarização de Respostas (Sim/Não) e Strings Numéricas
    print("🔄 Binarizando colunas de doenças e sintomas...")
    binary_map = {
    'SIM': 1, 
    'NÃO': 0, 
    'NÃ\x83O': 0,  # Necessário pra evitar erros
    '1': 1, 
    '0': 0}
    df_proc.replace(binary_map, inplace=True)

    # 2. One-Hot Encoding para Sexo
    print("🧬 Aplicando One-Hot Encoding em 'cs_sexo'...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Criamos o DataFrame encodado com nomes de colunas explícitos
    encoded_data = encoder.fit_transform(df_proc[['cs_sexo']])
    encoded_df = pd.DataFrame(
        encoded_data, 
        columns=['F', 'M'], 
        index=df_proc.index
    )
    
    # Substituição da coluna original pelas novas binárias
    df_proc = pd.concat([encoded_df, df_proc.drop(columns=['cs_sexo'])], axis=1)

    # Preservamos a idade sem o scaler para segmentação de grupo
    df_proc['idade_raw'] = df_proc['idade'].astype(int)

    # 3. Escalonamento da Idade (Crucial para a sensibilidade do modelo)
    print(f"📏 Aplicando MinMaxScaler na idade (Max original: {df_proc['idade'].max()})...")
    scaler = MinMaxScaler()
    df_proc['idade'] = scaler.fit_transform(df_proc[['idade']])

    # 4. Inspeção e Tratamento de Inconsistências (Mantendo seus logs de segurança)
    non_numeric = df_proc.select_dtypes(exclude=['number']).columns
    
    if len(non_numeric) > 0:
        print(f"⚠️ Alerta: Colunas com texto detectadas: {list(non_numeric)}")
        for col in non_numeric:
            exemplos = df_proc[col].unique()[:5]
            print(f"   - Coluna '{col}' contém valores inesperados como: {exemplos}")
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        
        df_proc.dropna(inplace=True)

    # 5. Otimização Final (Tipagem Híbrida para não destruir o Scaler)
    print("📉 Otimizando memória: float32 (idade) e int8 (binários)...")
    for col in df_proc.columns:
        if col == 'idade':
            df_proc[col] = df_proc[col].astype('float32')

        elif col == 'idade_raw':
            df_proc[col] = df_proc[col].astype('int16') #considerar trocar pra int8 futuramente

        else:
            # int8 ocupa apenas 1 byte por linha, ideal para 0 e 1
            df_proc[col] = df_proc[col].astype('int8') 

    print(f"✅ Pré-processamento finalizado. Shape: {df_proc.shape}")
    return df_proc



def export_data(df, data_dir='data'):
    """
    Exporta o DataFrame processado para CSV e exibe o tamanho do arquivo gerado.
    """
    if df is None:
        raise ValueError("❌ Erro: Não há dados para exportar.")
        
    output_path = os.path.join(data_dir, 'data_ready.csv')
    
    # Exportação
    df.to_csv(output_path, index=False)
    
    # Cálculo do tamanho do arquivo (em MB)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"💾 Dados exportados com sucesso para: {output_path}")
    print(f"📂 Tamanho do arquivo final: {file_size:.2f} MB")
    
    return output_path



def run_pipeline(data_dir='data', export=True):
    """
    Executa o fluxo completo. Se export=True, salva o resultado em disco.
    """
    print(f"🚀 Iniciando Pipeline no diretório: {data_dir}")
    
    # 1. Extração
    raw_path = extract(data_dir)
    
    # 2. Importação
    df_raw = import_data(raw_path)
    
    # 3. Limpeza
    df_cleaned = clean_data(df_raw)
    
    # 4. Pré-processamento
    df_final = pre_process(df_cleaned)
    
    # 5. Exportação (Nova etapa condicional)
    if export:
        export_data(df_final, data_dir)
    
    print("🏁 Pipeline finalizado.")
    return df_final


def test_data(data_dir='data', export=True):
    raw_path = extract(data_dir)
    df_raw = import_data(raw_path)

    cols_to_remove = [
        'data_inicio_sintomas', 'codigo_ibge', 
        'diagnostico_covid19', 'nome_munic'
    ]
    
    print("🧹 Iniciando limpeza dos dados...")

    # 2. Remoção de Colunas (Drop antecipado para liberar memória)
    df_cleaned = df.drop(columns=cols_to_remove).copy()
    print(f"   - {len(cols_to_remove)} colunas desnecessárias removidas.")

    return df_cleaned
