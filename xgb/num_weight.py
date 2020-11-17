import json


def findAll(target, dictData, notFound=[]):
    queue = [dictData]
    result = []
    while len(queue) > 0:
        data = queue.pop()
        for key, value in data.items():
            if key == target:
                result.append(value)
            elif type(value) == dict:
                queue.append(value)
    if not result:
        result = notFound
    return result


if __name__ == "__main__":
    size = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    for s in size:
        with open('model/xgb/6_{}.json'.format(s), 'r') as f:
            model_json = json.load(f)
        trees = findAll('trees', model_json)[0]
        num_base_weights = 0
        for node in trees:
            base_weights = node['base_weights']
            num_base_weights += len(base_weights)
        print('datasize : {}, num of all base weights'.format(s), num_base_weights)
