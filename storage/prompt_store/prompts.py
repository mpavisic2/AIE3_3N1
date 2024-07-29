

class N3Prompts:

    def __init__(self):
        # get customer_by oib prompt
        self.cid_qe_prompt = """
        Context information is below.\n
        ---------------------\n
        {context_str}\n
        ---------------------\n
        Given the context information and not prior knowledge, 
        find company data, naziv key needs to have company from full company name from query in part of the context co to give this information back
        Query: {query_str}\n
        answer always in json format with key:value pairs, if you don't find the answer respond with na
        ### Example:
        {{'oib':'2324324432', 'naziv':'Podravka d.d.'}}"""

        # sql query engine teplate
        self.sql_qe_template = {"query_tool_engine_desc":"""
    This tool should be used to answer question related to the mobile by translating a natural language query into a SQL query with access to tables, always try to use quries provided in examples:
    Search criteria should be always case insensitive, and use like in where clause
    'mobile_customer_base' - mobile customer base that return data about customers (oib) - number of subscribers - count(subscriber_id), fees_to_charge=number of months till contract expires, ARPA_2024 = arpa
    """,
        "query_tool_custom_prompt":"""
            Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.

            Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

            Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

            Question: Question here
            SQLQuery: SQL Query to run
            SQLResult: Result of the SQLQuery
            Answer: Final answer here

            Only use tables listed below.
            {schema}

            examples:
            Question: what customer has the most subscribers
            SQLQuery: SELECT oib, COUNT(subscriber_id) AS num_subscribers\nFROM mobile_customer_base\nGROUP BY oib\nORDER BY num_subscribers DESC\nLIMIT 1;
            Question: customer with the most subscribers with contract expiry less than 3
            SQLQuery: SELECT oib, COUNT(subscriber_id) AS num_subscribers\nFROM mobile_customer_base\nWHERE Fees_to_charge < 3\nGROUP BY oib\nORDER BY num_subscribers DESC\nLIMIT 1;

            Question: {query_str}
            SQLQuery: 
    """}
        # news search prompt

        self.news_search_prompt = { "user": {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": ""
            }
        ]
        },
        "system": {
        "role": "system",
        "content":  [
                {
                "type": "text",
                "text": """Sumiraj rezultate pretrage i sažmi ih u kratki sažetak od 5 rečenica. 
            Odogovor neka preferira informacije s hrvatskih news portala, a zatim i sa drugih hrvatskih portala. 
            Potraži vijesti o firmi i sažmi ih u kratki sažetak od 5 rečenica. 
            Vijesti neka imaju prednost u prikazu ispred osnovnih informacija o tvrtki.
            Nakon pretrage provjeri sadrže li razultati pretrage informacije o tvrtki.
            One rezultate koji ne sadrže, izostavi iz finalnog odgovora.

            Odgovor neka bude prema primjeru:

            1. Tvrtka Primjer je otvorila novi ured prije mjesec dana (jutarnji.hr)
            2. Istraga otvorene nakon smunji u korupciju u tvrtki Primjer (index.hr)

        
            """}]
        }
        }
        self.internal_360_id = { "user": {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": ""
            }
        ]
        },
        "system": {
        "role": "system",
        "content":  [
                {
                "type": "text",
                "text": """Ti si alat koji uljepšava tekst, formatiraj tablicu 1, a tablicu 2 pretovri u tekstualni oblik s opisom.
                        Ne spomiji termine tablica 1 i 2 u odgovoru, već im daj naziv prema podacima koje prikazuju.
                        Trendovi su prikazani u postotcima!
            """}]
        }
        }
        self.public_data_prompt = """Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Given the context information and not prior knowledge, 
            Find and summarize company public data\n
            Analyse company financial trend if data is provided in the context\n
            Query: {query_str}\n
            Answer in form of news article"""
        self.public_data_prompt_hr = """Niže je naveden kontektst.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            Korištenjem informacija iz konteksta, a ne tvog prethodnog znanja, 
            Pronađi javne podatke o kompaniji ukoliko je općenit upit o kompaniji\n
            U slučaju specifičnog upita potraži samo odgovor na njih\n
            Query: {query_str}\n
            Odgovori u formi natuknica"""
        self.customer_360_tool = {"sql_report_prompt":f"""
            find avg for arpa, voice usage, discount, count of subscribers and overshoot 
            by tariff_model for oib
            """}
        self.c_360_en = """ this tool gives customer 360 overview in Croatian language, input of tool is dict in form of e.g. {{'oib': '1111111111','naziv': 'Naziv firme'}}
        output of each part (public data, internal data, news) should not be longer than one 250 tokens """
        self.nnn_agent = """
            Ti si veseli asistent za managere! 
            
            Pružaš informacije na globalnoj razini ili prema OIB-u, ovisno o upitu.

            Ako OIB ili naziv nije dostupan u podacima, vratit ćeš 'nema informacija'.
            Ako je upit vezan uz određenu tvrtku, prvo koristiš company_name_and_id_finder za pronalaženje podataka o tvrtki, a zatim nastavljaš s daljnjim koracima!
            Ako upit sadrži OIB, prvo koristi funkciju oib2name, a zatim nastavi s daljnjim koracima!
            
            Unos za svaki alat trebao bi biti u obliku npr. {'oib':'1111111111', 'naziv':'Neka tvrtka', 'metapodaci':'direktor tvrtke'}
            
            sql_tool: Imaj na umu da su subscriber_id i customer_id ID kolone, a ne kvantitativne kolone! Uvijek koristi OIB kao unos za sql_tool.
            Ako upit traži 360, odgovori sirovim izlazom iz customer_360_view, ali ga formatiraj kao markdown.

            U slučaju pogreške, uvijek traži više informacija i predloži koje informacije možeš pružiti iz dostupnog seta alata.
            Konačan odgovor treba biti na hrvatskom jeziku.
            """
        self.nnn_agent_nmhr = """
            Ti si veseli asistent za managere! 
            
            Pružaš informacije na globalnoj razini ili prema OIB-u, ovisno o upitu.

            Ako OIB ili naziv nije dostupan u podacima, vratit ćeš 'nema informacija'.
            Ako je upit vezan uz određenu tvrtku, prvo koristiš company_name_and_id_finder za pronalaženje podataka o tvrtki, a zatim nastavljaš s daljnjim koracima!
            Ako upit sadrži OIB, prvo koristi funkciju oib2name, a zatim nastavi s daljnjim koracima!
            
            Unos za svaki alat trebao bi biti u obliku npr. {'oib':'1111111111', 'naziv':'Neka tvrtka', 'metapodaci':'direktor tvrtke'}
            
            sql_tool: Imaj na umu da su subscriber_id i customer_id ID kolone, a ne kvantitativne kolone! Uvijek koristi OIB kao unos za sql_tool.
            Ako upit traži 360, odgovori sirovim izlazom iz customer_360_view.

            U slučaju pogreške, uvijek traži više informacija i predloži koje informacije možeš pružiti iz dostupnog seta alata.
            """
        

    def get_sql_report_prompt(self, qry):
        self.customer_360_tool["sql_report_prompt"]=f"""
        Upit treba izračunati  prosjek overshoota , ukupan broj priključaka, prosjek voice usage-a, prosjek roaming usage koristeći group by tariff_model i filter po oib = {qry['oib']}
        ako korigiraš upit pokreni taj korigirani upit
        output funkcije moraju biti podatci iz baze, ne informacija o nemogućnosti dohvata podataka

"""
        #print(self.customer_360_tool)
        self.customer_360_tool["sql_report_prompt"] = """
1. **Broj priključaka u bazi za tvrtku:**
    ```sql
    SELECT count(subscriber_id) 
    FROM mobile_customer_base 
    where oib={oib}
    ```

        U odgovoru napiši sažetak ovih informacija uz napomenu kako je potrebno kontaktirati ove korisnike kojima ističe ugovor
        """

        return self.customer_360_tool["sql_report_prompt"]

    def get_news_search_prompt(self,web_resp):
        self.news_search_prompt["user"] = {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{web_resp}"
            }
        ]
        }

        return self.news_search_prompt["user"]
    
    def get_sql_360_prompt(self,str_input):
        self.internal_360_id["user"] = {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{str_input}"
            }
        ]
        }

        return self.internal_360_id["user"]
    
    



        

    

