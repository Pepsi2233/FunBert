""""
    接收一个csv文件路径列表。依次读取每个文件Sequence列最后200行数据seq1和
    倒数200-400行数据seq2和剩余数据seq3，将seq1追加保存到新建的test.txt文件中，
    将seq2追加保存在dev.txt，将seq3追加保存在train.txt文件中，其中test.txt、
    dev.txt和train.txt在同一文件夹下并且该文件夹由用户定义
"""
import math

import pandas as pd
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_kmer_sentence(original_string, kmer=1, stride=1):
    if kmer == -1:
        return original_string

    sentence = ""
    original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string) - kmer:
        sentence += original_string[i:i + kmer] + " "
        i += stride

    return sentence[:-1].strip("\"")


def process_csv_files(input_folder,csv_file_paths, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 设置index为类别数
    index = 0
    for csv_file_path in csv_file_paths:
        # 使用os.path.join拼接路径并检查路径是否存在
        file_path = os.path.join(input_folder, csv_file_path)
        print(file_path)
        if os.path.exists(file_path):
            print("文件存在")
        else:
            print("文件不存在")

        # 读取CSV文件
        df = pd.read_csv(file_path)

        j = math.floor(len(df['Sequence']) * 0.8)
        k = j + math.floor(len(df['Sequence']) * 0.1)
        m = k+ math.floor(len(df['Sequence']) * 0.1)
        print(len(df['Sequence']), j, k, m)
        # 提取不同部分的数据,写入不同的文本文件
        with open(os.path.join(output_folder, 'train.txt'), 'a') as train_txt:
            headers1 = []
            for i in range(0, j):
                # seq1为第i条数据
                headers1.append(df['Header'][i]+ ";" +df['Sequence'][i])
                seq1 = df['Sequence'][i]
                if len(seq1) >= 514:
                    sequence = get_kmer_sentence(seq1[:514], 6, 1)+"*"+str(index)
                    train_txt.write(sequence + '\n')
                else:
                    sequence = get_kmer_sentence(seq1[0:], 6, 1) + "*" + str(index)
                    train_txt.write(sequence + '\n')
        with open(os.path.join(output_folder, 'Compare_train.txt'), mode='a', newline='') as file:
            for item in headers1:
                file.write(item)
                file.write('\n')


        with open(os.path.join(output_folder, 'dev.txt'), 'a') as dev_file:
            headers2 = []
            for i in range(j, k):
                headers2.append(df['Header'][i] + ";" + df['Sequence'][i])
                seq2 = df['Sequence'][i]
                if len(seq2) >= 514:
                    sequence = get_kmer_sentence(seq2[:514], 6, 1) + "*" + str(index)
                    dev_file.write(sequence + '\n')
                else:
                    sequence = get_kmer_sentence(seq2[0:], 6, 1) + "*" + str(index)
                    dev_file.write(sequence + '\n')

        with open(os.path.join(output_folder, 'Compare_test.txt'), mode='a', newline='') as file:
            for item in headers2:
                file.write(item)
                file.write('\n')

        with open(os.path.join(output_folder, 'test.txt'), 'a') as test_file:
            headers3 = []
            for i in range(k, m):
                seq3 = df['Sequence'][i]
                headers3.append(df['Header'][i] + ";" + df['Sequence'][i])
                if len(seq3) >= 514:
                    sequence = get_kmer_sentence(seq3[:514], 6, 1) + "*" + str(index)
                    test_file.write(sequence + '\n')
                else:
                    sequence = get_kmer_sentence(seq3[0:], 6, 1) + "*" + str(index)
                    test_file.write(sequence + '\n')

        with open(os.path.join(output_folder, 'Compare_test.txt'), mode='a', newline='') as file:
            for item in headers3:
                file.write(item)
                file.write('\n')

        # 类别数+1
        index += 1
    print("类别数：{0}".format(index))


def shuffle_txt_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是 txt 文件
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # 打开文本文件并读取内容
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # 打乱内容顺序
            random.shuffle(lines)
            # 覆盖原文件内容
            with open(file_path, 'w') as file:
                file.writelines(lines)


# 文件夹前缀
input_folder = "../Creature_Level/(54)Genus"
# 这是输入已经分好类之后的csv文件名列表，对他们进行
csv_files = ['Plenodomus.csv', 'Phyllactinia.csv', 'Athelia.csv', 'Gymnopus.csv', 'Torula.csv', 'Apiospora.csv', 'Hyphopichia.csv', 'Camarophyllopsis.csv', 'Cryptotrichosporon.csv', 'Seimatosporium.csv', 'Rhizocarpon.csv', 'Hyaloscypha.csv', 'Lasiodiplodia.csv', 'Mycenella.csv', 'Lachnum.csv', 'Pachyphlodes.csv', 'Helicosporium.csv', 'Kazachstania.csv', 'Lobaria.csv', 'Funneliformis.csv', 'Cytospora.csv', 'Marasmius.csv', 'Castanediella.csv', 'Alatospora.csv', 'Umbelopsis.csv', 'Pleurotus.csv', 'Coccocarpia.csv', 'Plagiostoma.csv', 'Venturia.csv', 'Melanconiella.csv', 'Pulveroboletus.csv', 'Erysiphe.csv', 'Rosellinia.csv', 'Pluteus.csv', 'Montagnula.csv', 'Exobasidium.csv', 'Lophodermium.csv', 'Acarospora.csv', 'Pseudocyphellaria.csv', 'Sugiyamaella.csv', 'Octaviania.csv', 'Hygrophorus.csv', 'Hemimycena.csv', 'Micropsalliota.csv', 'Ceriporia.csv', 'Xerocomellus.csv', 'Helvellosebacina.csv', 'Peniophorella.csv', 'Rhizoctonia.csv', 'Cladia.csv', 'Lecanora.csv', 'Devriesia.csv', 'Trametes.csv', 'Toxicocladosporium.csv', 'Filobasidium.csv', 'Leucocoprinus.csv', 'Phyllopsora.csv', 'Thozetella.csv', 'Golovinomyces.csv', 'Falciformispora.csv', 'Saccharomycopsis.csv', 'Cryptococcus.csv', 'Leccinum.csv', 'Leucoagaricus.csv', 'Sporidesmium.csv', 'Galerina.csv', 'Resupinatus.csv', 'Cylindrosympodium.csv', 'Geomyces.csv', 'Ochroconis.csv', 'Cyathus.csv', 'Kwoniella.csv', 'Paraconiothyrium.csv', 'Puccinia.csv', 'Diaporthe.csv', 'Cyphellophora.csv', 'Idriella.csv', 'Cadophora.csv', 'Hypholoma.csv', 'Orbilia.csv', 'Tricholoma.csv', 'Inosperma.csv', 'Rhodocollybia.csv', 'Gaeumannomyces.csv', 'Infundichalara.csv', 'Roccella.csv', 'Mortierella.csv', 'Psathyrella.csv', 'Kockovaella.csv', 'Candelabrum.csv', 'Auricularia.csv', 'Hymenoscyphus.csv', 'Lentinellus.csv', 'Sclerococcum.csv', 'Kavinia.csv', 'Cyberlindnera.csv', 'Tricladium.csv', 'Phaeophyscia.csv', 'Circinaria.csv', 'Biscogniauxia.csv', 'Tulasnella.csv', 'Glutinoglossum.csv', 'Umbilicaria.csv', 'Cistella.csv', 'Sarcodon.csv', 'Odontia.csv', 'Phialocephala.csv', 'Protoparmelia.csv', 'Helicogloea.csv', 'Delastria.csv', 'Lepiota.csv', 'Chaetosphaeria.csv', 'Lycoperdon.csv', 'Arrhenia.csv', 'Peziza.csv', 'Aureoboletus.csv', 'Debaryomyces.csv', 'Gnomoniopsis.csv', 'Hemileucoglossum.csv', 'Cortinarius.csv', 'Hohenbuehelia.csv', 'Dirina.csv', 'Amauroascus.csv', 'Trechispora.csv', 'Pisolithus.csv', 'Marasmiellus.csv', 'Tephrocybe.csv', 'Exserohilum.csv', 'Xylaria.csv', 'Derxomyces.csv', 'Cantharellus.csv', 'Phomatospora.csv', 'Butyriboletus.csv', 'Gongronella.csv', 'Oidiodendron.csv', 'Polyscytalum.csv', 'Stagonospora.csv', 'Dacrymyces.csv', 'Conioscypha.csv', 'Ustilago.csv', 'Ascosphaera.csv', 'Pestalotiopsis.csv', 'Lophiostoma.csv', 'Apiosordaria.csv', 'Pseudosperma.csv', 'Thaxterogaster.csv', 'Lepraria.csv', 'Hypotrachyna.csv', 'Paraphoma.csv', 'Morchella.csv', 'Calogaya.csv', 'Antrodia.csv', 'Hyphodiscus.csv', 'Simplicillium.csv', 'Ramophialophora.csv', 'Postia.csv', 'Cenococcum.csv', 'Pulvinula.csv', 'Ramaria.csv', 'Auxarthron.csv', 'Phlebiopsis.csv', 'Trichoglossum.csv', 'Tilletia.csv', 'Metarhizium.csv', 'Dictyochaeta.csv', 'Yamadazyma.csv', 'Lyomyces.csv', 'Crepidotus.csv', 'Aspergillus.csv', 'Calonarius.csv', 'Monosporascus.csv', 'Nephroma.csv', 'Membranomyces.csv', 'Heterodermia.csv', 'Diversispora.csv', 'Meliniomyces.csv', 'Phaeosphaeriopsis.csv', 'Agrocybe.csv', 'Rutstroemia.csv', 'Cordyceps.csv', 'Protomerulius.csv', 'Geoglossum.csv', 'Bovista.csv', 'Albatrellus.csv', 'Hymenogaster.csv', 'Wickerhamomyces.csv', 'Leptographium.csv', 'Solicoccozyma.csv', 'Ramariopsis.csv', 'Ramularia.csv', 'Coniophora.csv', 'Scytalidium.csv', 'Dinemasporium.csv', 'Peltigera.csv', 'Caloplaca.csv', 'Chlorophyllum.csv', 'Gorgomyces.csv', 'Polyozellus.csv', 'Eutypella.csv', 'Hannaella.csv', 'Lecythophora.csv', 'Mycoleptodiscus.csv', 'Paecilomyces.csv', 'Diploschistes.csv', 'Lentinus.csv', 'Rhizophagus.csv', 'Neofusicoccum.csv', 'Ophiosphaerella.csv', 'Lichtheimia.csv', 'Eocronartium.csv', 'Hysterangium.csv', 'Wiesneriomyces.csv', 'Thelephora.csv', 'Cladosporium.csv', 'Syncephalis.csv', 'Hebeloma.csv', 'Microdochium.csv', 'Peniophora.csv', 'Waitea.csv', 'Entoloma.csv', 'Bjerkandera.csv', 'Spadicoides.csv', 'Tubeufia.csv', 'Pseudocercospora.csv', 'Phellinus.csv', 'Tylopilus.csv', 'Rigidoporus.csv', 'Powellomyces.csv', 'Rhizopogon.csv', 'Pyrenophora.csv', 'Coprinellus.csv', 'Dermoloma.csv', 'Cryptosporiopsis.csv', 'Crocicreas.csv', 'Tulostoma.csv', 'Cladorrhinum.csv', 'Basidiodendron.csv', 'Pholiota.csv', 'Tolypocladium.csv', 'Myrothecium.csv', 'Cladonia.csv', 'Kondoa.csv', 'Subulicystidium.csv', 'Rinodina.csv', 'Periconia.csv', 'Virgaria.csv', 'Phaeocollybia.csv', 'Sebacina.csv', 'Podospora.csv', 'Bipolaris.csv', 'Arthroderma.csv', 'Seiridium.csv', 'Pseudodactylaria.csv', 'Claroideoglomus.csv', 'Dothiora.csv', 'Colletotrichum.csv', 'Phoma.csv', 'Pseudophialophora.csv', 'Ruhlandiella.csv', 'Stereum.csv', 'Podoscypha.csv', 'Zasmidium.csv', 'Fomitiporia.csv', 'Phylloporia.csv', 'Lyophyllum.csv', 'Craterellus.csv', 'Helvella.csv', 'Teratosphaeria.csv', 'Scutellospora.csv', 'Papiliotrema.csv', 'Anthracocystis.csv', 'Verruconis.csv', 'Hymenochaete.csv', 'Arnium.csv', 'Arthrobotrys.csv', 'Cordana.csv', 'Phallus.csv', 'Beauveria.csv', 'Pervetustus.csv', 'Melanogaster.csv', 'Rhizoglomus.csv', 'Leptosphaeria.csv', 'Claviceps.csv', 'Aspicilia.csv', 'Usnea.csv', 'Lactarius.csv', 'Gymnopilus.csv', 'Armillaria.csv', 'Tuber.csv', 'Vararia.csv', 'Spirosphaera.csv', 'Neobulgaria.csv', 'Mycosphaerella.csv', 'Gigaspora.csv', 'Rectipilus.csv', 'Protoparmeliopsis.csv', 'Lecidea.csv', 'Pyrenochaeta.csv', 'Pseudotomentella.csv', 'Arthrocladium.csv', 'Gyalolechia.csv', 'Chaetomium.csv', 'Geosmithia.csv', 'Hawksworthiomyces.csv', 'Microascus.csv', 'Stereocaulon.csv', 'Strobilomyces.csv', 'Niesslia.csv', 'Coniella.csv', 'Rhizoplaca.csv', 'Sporormiella.csv', 'Tubaria.csv', 'Stropharia.csv', 'Blastobotrys.csv', 'Humaria.csv', 'Crinipellis.csv', 'Hypomyces.csv', 'Melanophyllum.csv', 'Preussia.csv', 'Scytinostroma.csv', 'Cladophialophora.csv', 'Clavispora.csv', 'Occultifur.csv', 'Elsinoe.csv', 'Roussoella.csv', 'Aleurodiscus.csv', 'Distoseptispora.csv', 'Rhinocladiella.csv', 'Backusella.csv', 'Tubulicrinis.csv', 'Glomus.csv', 'Rhizopus.csv', 'Monographella.csv', 'Sticta.csv', 'Phlebia.csv', 'Phaeotremella.csv', 'Acaulospora.csv', 'Tremella.csv', 'Parmelia.csv', 'Dactylospora.csv', 'Paracremonium.csv', 'Fusicolla.csv', 'Elaphomyces.csv', 'Chroogomphus.csv', 'Pterula.csv', 'Hydropus.csv', 'Volvariella.csv', 'Clonostachys.csv', 'Tephromela.csv', 'Thecaphora.csv', 'Laetisaria.csv', 'Basidiobolus.csv', 'Fuscoporia.csv', 'Neonectria.csv', 'Spathaspora.csv', 'Pyrenochaetopsis.csv', 'Scheffersomyces.csv', 'Codinaea.csv', 'Scleroderma.csv', 'Spizellomyces.csv', 'Inonotus.csv', 'Dioszegia.csv', 'Phlegmacium.csv', 'Gerronema.csv', 'Clavariadelphus.csv', 'Lopadostoma.csv', 'Porpidia.csv', 'Sparassis.csv', 'Candelariella.csv', 'Scutellinia.csv', 'Terfezia.csv', 'Bacidia.csv', 'Hanseniaspora.csv', 'Westerdykella.csv', 'Paraphaeosphaeria.csv', 'Nemania.csv', 'Phellodon.csv', 'Cristinia.csv', 'Gibellulopsis.csv', 'Plectania.csv', 'Ramalina.csv', 'Archaeospora.csv', 'Cyanosporus.csv', 'Laetiporus.csv', 'Dominikia.csv', 'Clavaria.csv', 'Calcarisporiella.csv', 'Vishniacozyma.csv', 'Articulospora.csv', 'Sympodiella.csv', 'Tomentellopsis.csv', 'Monocillium.csv', 'Parasola.csv', 'Buellia.csv', 'Laccaria.csv', 'Acrocalymma.csv', 'Rachicladosporium.csv', 'Amauroderma.csv', 'Chlorociboria.csv', 'Aureobasidium.csv', 'Trichoderma.csv', 'Pseudeurotium.csv', 'Nigrospora.csv', 'Phaeoacremonium.csv', 'Pseudosigmoidea.csv', 'Trichophaea.csv', 'Pezicula.csv', 'Helminthosporium.csv', 'Xanthoparmelia.csv', 'Picoa.csv', 'Neocamarosporium.csv', 'Chrysosporium.csv', 'Curvularia.csv', 'Pertusaria.csv', 'Pectenia.csv', 'Physcia.csv', 'Hypoxylon.csv', 'Melanohalea.csv', 'Gliophorus.csv', 'Clavulinopsis.csv', 'Sclerostagonospora.csv', 'Fulvifomes.csv', 'Psora.csv', 'Coccomyces.csv', 'Annulohypoxylon.csv', 'Ganoderma.csv', 'Phaeococcomyces.csv', 'Endocarpon.csv', 'Trichomerium.csv', 'Rhytidhysteron.csv', 'Pycnora.csv', 'Microdominikia.csv', 'Scolecobasidium.csv', 'Parastagonospora.csv', 'Dictyonema.csv', 'Dendriscosticta.csv', 'Rasamsonia.csv', 'Thelonectria.csv', 'Sistotrema.csv', 'Balsamia.csv', 'Geotrichum.csv', 'Rhizophydium.csv', 'Thanatephorus.csv', 'Cunninghamella.csv', 'Apiotrichum.csv', 'Scleropezicula.csv', 'Cladobotryum.csv', 'Dactylella.csv', 'Coltricia.csv', 'Parmotrema.csv', 'Massarina.csv', 'Genea.csv', 'Cercophora.csv', 'Inocybe.csv', 'Xanthoria.csv', 'Gyroporus.csv', 'Saitozyma.csv', 'Lecanicillium.csv', 'Phyllachora.csv', 'Hydnotrya.csv', 'Uromyces.csv', 'Entorrhiza.csv', 'Botryosphaeria.csv', 'Cystobasidium.csv', 'Leptogium.csv', 'Stephanospora.csv', 'Amanita.csv', 'Typhula.csv', 'Rhodotorula.csv', 'Cephaliophora.csv', 'Geminibasidium.csv', 'Suillus.csv', 'Talaromyces.csv', 'Kochiomyces.csv', 'Arachnopeziza.csv', 'Saksenaea.csv', 'Tomentella.csv', 'Kamienskia.csv', 'Coltriciella.csv', 'Clitocybula.csv', 'Geopora.csv', 'Helicoma.csv', 'Humicola.csv', 'Pichia.csv', 'Exophiala.csv', 'Lindtneria.csv', 'Phaeoclavulina.csv', 'Polyporus.csv', 'Blastenia.csv', 'Coniosporium.csv', 'Kretzschmaria.csv', 'Mytilinidion.csv', 'Schizothecium.csv', 'Volutella.csv', 'Hyphoderma.csv', 'Myxozyma.csv', 'Hydnobolites.csv', 'Disciseda.csv', 'Exidia.csv', 'Trogia.csv', 'Tricholomopsis.csv', 'Tarzetta.csv', 'Herpotrichia.csv', 'Piloderma.csv', 'Gyromitra.csv', 'Entyloma.csv', 'Alternaria.csv', 'Cerinomyces.csv', 'Neodevriesia.csv', 'Ophiognomonia.csv', 'Eutypa.csv', 'Setophoma.csv', 'Rhodosporidiobolus.csv', 'Melanoleuca.csv', 'Vermiculariopsiella.csv', 'Daldinia.csv', 'Xylodon.csv', 'Arthrinium.csv', 'Echinoderma.csv', 'Piskurozyma.csv', 'Ogataea.csv', 'Anthostomella.csv', 'Glutinomyces.csv', 'Byssocorticium.csv', 'Pyrenula.csv', 'Cosmospora.csv', 'Penicillium.csv', 'Verticillium.csv', 'Tricharina.csv', 'Hydnum.csv', 'Cystoderma.csv', 'Coleophoma.csv', 'Stachybotrys.csv', 'Didymella.csv', 'Nigrograna.csv', 'Naganishia.csv', 'Veronaeopsis.csv', 'Mallocybe.csv', 'Gautieria.csv', 'Capronia.csv', 'Leucosporidium.csv', 'Ambispora.csv', 'Taphrina.csv', 'Tetragoniomyces.csv', 'Coniothyrium.csv', 'Nectria.csv', 'Mollisia.csv', 'Limonomyces.csv', 'Conocybe.csv', 'Micarea.csv', 'Hydnellum.csv', 'Microbotryum.csv', 'Phanerochaete.csv', 'Wickerhamiella.csv', 'Siepmannia.csv', 'Minutisphaera.csv', 'Dermatocarpon.csv', 'Hirsutella.csv', 'Zopfiella.csv', 'Deconica.csv', 'Tumularia.csv', 'Cephalotheca.csv', 'Dentiscutata.csv', 'Myxarium.csv', 'Mucor.csv', 'Trichophyton.csv', 'Polyblastia.csv', 'Septoria.csv', 'Tetracladium.csv', 'Zygosaccharomyces.csv', 'Pseudoanungitea.csv', 'Lactifluus.csv', 'Septoriella.csv', 'Calicium.csv', 'Cryptodiscus.csv', 'Iodophanus.csv', 'Cystolepiota.csv', 'Cutaneotrichosporon.csv', 'Keissleriella.csv', 'Botryobasidium.csv', 'Physconia.csv', 'Melampsora.csv', 'Podosphaera.csv', 'Hymenopellis.csv', 'Leptodontidium.csv', 'Macrolepiota.csv', 'Clavulina.csv', 'Malassezia.csv', 'Xenopolyscytalum.csv', 'Mycena.csv', 'Hypogymnia.csv', 'Otidea.csv', 'Coprinopsis.csv', 'Myrmecridium.csv', 'Septoglomus.csv', 'Hypochnicium.csv', 'Lipomyces.csv', 'Saturnispora.csv', 'Nanoglomus.csv', 'Skeletocutis.csv', 'Gyrothrix.csv', 'Rhizophlyctis.csv', 'Phaeohelotium.csv', 'Chaenotheca.csv', 'Ramicandelaber.csv', 'Uromycladium.csv', 'Ceratobasidium.csv', 'Agaricus.csv', 'Serendipita.csv', 'Chondrogaster.csv', 'Steccherinum.csv', 'Bryoria.csv', 'Lecidella.csv', 'Lecania.csv', 'Hodophilus.csv', 'Conlarium.csv', 'Phialemonium.csv', 'Amphinema.csv', 'Pezoloma.csv', 'Clitopilus.csv', 'Fusarium.csv', 'Biatora.csv', 'Metschnikowia.csv', 'Epicoccum.csv', 'Dothiorella.csv', 'Dipodascus.csv', 'Phylloporus.csv', 'Ceratocystis.csv', 'Cuphophyllus.csv', 'Pseudoplectania.csv', 'Tetrapyrgos.csv', 'Betamyces.csv', 'Ascobolus.csv', 'Knufia.csv', 'Veronaea.csv', 'Sporobolomyces.csv', 'Russula.csv', 'Microglossum.csv', 'Clitocybe.csv', 'Xerocomus.csv', 'Perenniporia.csv', 'Cora.csv', 'Piptocephalis.csv', 'Phyllosticta.csv', 'Verrucaria.csv', 'Coniochaeta.csv', 'Sakaguchia.csv', 'Magnaporthiopsis.csv', 'Boletus.csv', 'Phaeosphaeria.csv', 'Ophiostoma.csv', 'Arachnomyces.csv', 'Calvatia.csv', 'Chloridium.csv', 'Stagonosporopsis.csv', 'Hyphodontia.csv', 'Hyalorbilia.csv', 'Harknessia.csv', 'Lachnellula.csv', 'Archaeorhizomyces.csv', 'Neosetophoma.csv', 'Sporothrix.csv', 'Sonoraphlyctis.csv', 'Ophiocordyceps.csv', 'Pleurotheciella.csv', 'Hygrocybe.csv', 'Agonimia.csv', 'Paraglomus.csv', 'Simocybe.csv', 'Rhodocybe.csv', 'Termitomyces.csv', 'Claussenomyces.csv', 'Boubovia.csv', 'Conidiobolus.csv', 'Geastrum.csv', 'Psilocybe.csv', 'Absidia.csv']

# 输出文件夹路径
output_folder = '(54)GenusCompare677'
# output_folder = '(54)Family307(508)(acc_)(more30)'
process_csv_files(input_folder, csv_files, output_folder)
# 打乱文件内容
shuffle_txt_files(output_folder)


