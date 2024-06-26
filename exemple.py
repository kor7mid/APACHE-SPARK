from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("demo").getOrCreate()

df = spark.createDataFrame(
    [
        ("sue", 32),
        ("li", 3),
        ("bob", 75),
        ("heo", 13),
    ],
    ["first_name", "age"],
)

df.show()
##============================================

# Ajoutons une colonne life_stage au DataFrame qui renvoie « enfant » si l'âge est de 12 ans ou moins, « adolescent » si l'âge est compris entre 13 et 19 ans et « adulte » si l'âge est de 20 ans ou plus.

from pyspark.sql.functions import col, when

df1 = df.withColumn(
    "life_stage",
    when(col("age") < 13, "child")
    .when(col("age").between(13, 19), "teenager")
    .otherwise("adult"),
)

df1.show()

##===========================================

# Les opérations Spark ne modifient pas le DataFrame. Vous devez affecter le résultat à une nouvelle variable pour accéder aux modifications DataFrame pour les opérations ultérieures.

##===========================================


# Filtrer un DataFrame Spark Maintenant, filtrez le DataFrame pour qu’il n’inclue que les adolescents et les adultes.

df1.where(col("life_stage").isin(["teenager", "adult"])).show()

##===========================================
# Regrouper par agrégation sur Spark DataFrame Calculons maintenant l’âge moyen de toutes les personnes figurant dans l’ensemble de données :

from pyspark.sql.functions import avg

df1.select(avg("age")).show()

# Vous pouvez également calculer l'âge moyen pour chaque étape de la vie :

df1.groupBy("life_stage").avg().show()

##===========================================
# Spark vous permet d'exécuter des requêtes sur des DataFrames avec SQL si vous ne souhaitez pas utiliser les API programmatiques. Interroger le DataFrame avec SQL Voici comment calculer l’âge moyen de chaque personne avec SQL :

spark.sql("select avg(age) from {df1}", df1=df1).show()

# Et voici comment calculer l’âge moyen par life_stage avec SQL :

spark.sql("select life_stage, avg(age) from {df1} group by life_stage", df1=df1).show()
##===========================================

# Spark vous permet d'utiliser l'API programmatique, l'API SQL ou une combinaison des deux. Cette flexibilité rend Spark accessible à une variété d’utilisateurs et puissamment expressif.

# Conservons le DataFrame dans une table nommée Parquet qui est facilement accessible via l'API SQL.

df1.write.saveAsTable("some_people")

# Assurez-vous que la table est accessible via le nom de la table :

spark.sql("select * from some_people").show()

# Maintenant, utilisons SQL pour insérer quelques lignes de données supplémentaires dans la table :

spark.sql("INSERT INTO some_people VALUES ('frank', 4, 'child')")

# Inspectez le contenu du tableau pour confirmer que la ligne a été insérée :

spark.sql("select * from some_people").show()

# Exécutez une requête qui renvoie les adolescents :

spark.sql("select * from some_people where life_stage='teenager'").show()

# Spark facilite l'enregistrement des tables et leur interrogation avec du SQL pur.
##===========================================
# Exemple de Spark RDD

# Les API Spark RDD conviennent aux données non structurées. L'API Spark DataFrame est plus simple et plus performante pour les données structurées. Supposons que vous ayez un fichier texte appelé some_text.txt avec les trois lignes de données suivantes :

# {these are words
# these are more words
# words in english} in some_text.txt file

# Vous souhaitez calculer le nombre de chaque mot dans le fichier texte. Voici comment effectuer ce calcul avec les Spark RDD :

text_file = spark.sparkContext.textFile("some_words.txt")

counts = (
    text_file.flatMap(lambda line: line.split(" "))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
)

# Jetons un coup d'œil au résultat 

counts.collect()