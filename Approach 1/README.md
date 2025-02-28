# How to use this

For symmetric and robust run this command:
```sh
./run.sh --loss robust --symmetric_noise 0.2 --rounds 100 --num_clients 10 --folder outputs
```
for symmetric and standard ce:
```sh
./run.sh --loss ce --symmetric_noise 0.2 --rounds 100 --num_clients 10 --folder outputs
```

similarly:

For assymetric and robust:
```sh
./run.sh --loss robust --asymmetric_noise 0.2 --rounds 100 --num_clients 10 --folder outputs
```

For assymetric and standard ce:
```sh
./run.sh --loss ce --asymmetric_noise 0.2 --rounds 100 --num_clients 10 --folder outputs
```
