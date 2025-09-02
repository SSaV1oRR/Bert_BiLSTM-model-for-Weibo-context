import random

dataset = 'dataset2'  # 数据集

def ReadFileDatas():
    FileNamelist = []
    file = open(dataset+'/data/alldatas2.txt', 'r+', encoding='utf_8_sig')
    for line in file:
        line = line.strip('\n')  # 删除每一行的\n
        FileNamelist.append(line)
    print('len ( FileNamelist ) = ', len(FileNamelist))
    file.close()
    return FileNamelist


def WriteDatasToFile(listInfo):
    file_handle = open(dataset+'/data/alldatas_random.txt', mode='w', encoding='utf_8_sig')
    for idx in range(len(listInfo)):
        str = listInfo[idx]
        file_handle.write(str+'\n')
    file_handle.close()


if __name__ == "__main__":
    listFileInfo = ReadFileDatas()
    # 打乱列表中的顺序
    random.shuffle(listFileInfo)
    WriteDatasToFile(listFileInfo)
