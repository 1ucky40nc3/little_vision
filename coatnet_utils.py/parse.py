with open("logs_coatnet.txt", "r") as f:
    lines = f.readlines()

print(len(lines))


eval_acc = list(filter(lambda x: "INFO  * Acc@1" in x, lines))
eval_acc = eval_acc[9:]
eval_steps = []

for line in eval_acc:
    values = line.split("Acc@1")[1].split("Acc@5")
    step = []
    for value in values:
        value = value.replace("\n", "").replace(" ", "")
        step.append(float(value))
    eval_steps.append(step)

print(len(eval_steps))
print(len(eval_steps[0]))
print(eval_steps[:30])

eval_loss = []
for line in eval_acc:
    idx = lines.index(line)
    before = lines[idx - 1]
    elem = before.split("Loss")[1].split("Acc@1")[0]
    elem = elem.split("(")[1].split(")")[0]
    elem = elem.replace("\n", "").replace(" ", "")
    eval_loss.append(float(elem))

print(len(eval_loss))
print(eval_loss)

"""
train = list(filter(lambda x: "INFO Train: [" in x, lines))
print(len(train))

train_steps = []
for line in train:
    split = line.split("loss")
    epoch_value = split[0].split("Train: [")[1].split("/")[0]
    loss_value = split[1].split("(")[0]
    value = value.replace("\n", "").replace(" ", "")
    train_steps.append((int(epoch_value), float(value)))

len(train_steps)
"""
import wandb

wandb.init(project="pytorch-image-models")

for i, (v, l) in enumerate(zip(eval_steps, eval_loss)):
    top1, top5 = v
    wandb.log(dict(epoch=i, eval_top1=top1, eval_top5=top5, eval_loss=l))



