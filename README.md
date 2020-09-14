## Challenge Homepage
https://www.drivendata.org/competitions/63/genetic-engineering-attribution/


## Discussion
https://community.drivendata.org/c/genetic-engineering-attribution/36


## BLAST Tutorial

1. download BLAST local: ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

or in Linux

```bash
sudo apt-get install ncbi-blast+
```

2. download and extract the Database: https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/

3. Download taxonemy 

(taxonemy database doesnt seem to work)

download taxid mapping file

ftp://ftp.ncbi.nih.gov/pub/taxonomy/accession2taxid/

```bash
sed '1d' prot.accession2taxid | awk '{print $2" "$3}' > accession_taxonid
```

```bash
update_blastdb taxdb
```
unzip it into the same folder of db files

add 
```bash
BLASTDB=/media/ac/BLAST
```
into .bashrc or create a .ncbirc file at the HOME dir.

4. Build Database: 

```bash 
makeblastdb -in nt -parse_seqids -dbtype nucl -out nt -taxid_map accession_taxonid
```

5. Run alignment

```bash
blastn -db nt -query test_seqs_group_0.fasta -out test.txt -num_threads 15 -outfmt "6 qseqid sseqid pident length mismatch gapopen sstart send evalue staxids sscinames sblastnames stitle" -num_alignments 1
```

https://www.tutorialspoint.com/biopython/biopython_overview_of_blast.htm