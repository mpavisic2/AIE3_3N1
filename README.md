Of course! Here is the translated specification for the application, with the name "3n1" or "Neural Nomad Nexus 1":

### Application Specification for "3n1" (Neural Nomad Nexus 1)

![](public/Picture1.png)

#### 1. Retrieving Company Information

**Question:** "Can you find information about the company XYZ d.o.o.?"

**Description:**
- We use the `functions.company_name_and_id_finder` tool to find the OIB and company name.
- Once we have the OIB, we use tools to search for public information, internal data, and news about the company to provide a more detailed overview.

**Query:**
```json
{
  "input": "XYZ d.o.o."
}
```

**Possible Answer:**
```plaintext
The company XYZ d.o.o. (OIB: 12345678901) was founded in 1995 and is engaged in the production of electronic devices. Recent news includes the opening of a new production facility. (jutarnji.hr)
```

#### 2. Calculating Average Overshoot and Voice Usage

**Question:** "What is the average overshoot and average voice usage for the company with OIB 1111111111?"

**Description:**
- We use the `functions.internal_data_tool` to calculate the average overshoot and voice usage for the specified OIB.

**Query:**
```json
{
  "qry": {
    "oib": "1111111111"
  }
}
```

**Possible Answer:**
```plaintext
The average overshoot for the company with OIB 1111111111 is 3.5 GB, while the average voice usage is 250 minutes per month.
```

#### 3. 360-Degree View of a Company

**Question:** "Can you show me a 360-degree view of the company with OIB 2222222222?"

**Description:**
- We use the `functions.customer_360_new` tool to provide a comprehensive overview of the company, including public information, internal data, and the latest news.

**Query:**
```json
{
  "qry": {
    "oib": "2222222222"
  }
}
```

**Possible Answer:**
```markdown
### 360-Degree View of the Company

**Public Information:**
- Company: ABC d.o.o.
- OIB: 2222222222
- Address: Ulica 123, Zagreb

**Internal Information:**
- Average overshoot: 4.2 GB
- Average voice usage: 300 minutes

**Latest News:**
1. The company ABC d.o.o. won an innovation award. (index.hr)
2. Opened a new office in Split. (vecernji.hr)
```

#### 4. Data on Mobile Subscribers

**Question:** "What are the data on mobile subscribers for the company with OIB 3333333333?"

**Description:**
- We use the `functions.sql-query` tool to retrieve data on mobile subscribers for the specified OIB.

**Query:**
```json
{
  "input": "SELECT count(subscriber_id), fees_to_charge, ARPA_2024 FROM mobile_customer_base WHERE oib = '3333333333'"
}
```

**Possible Answer:**
```plaintext
The company with OIB 3333333333 has 150 mobile subscribers. The average number of months until contract expiration is 6, and the ARPA for 2024 is 500 HRK.
```

#### 5. Searching for the Latest News about a Company

**Question:** "Can you find the latest news about the company DEF d.o.o.?"

**Description:**
- We use the `functions.search_company_news` tool to search for the latest news about the specified company.

**Query:**
```json
{
  "query": "DEF d.o.o."
}
```

**Possible Answer:**
```plaintext
1. The company DEF d.o.o. has launched a new product in the household appliances category. (dnevnik.hr)
2. DEF d.o.o. has announced a salary increase for all employees. (poslovni.hr)
```

### Conclusion

This application utilizes a range of specialized tools for retrieving and analyzing company data. Each tool is designed for a specific function, enabling detailed and accurate data analysis. If you have additional questions or need customization, feel free to reach out!