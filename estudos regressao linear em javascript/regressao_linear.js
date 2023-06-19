let x_vals = []; //declarando variavel local, lista
let y_vals = []; //declarando variavel local, lista

let m,b; //declarando variaveis locais

const learningRate = 0.5; //declarando uma constante de taxa de aprendizagem de valor 0.5
const optimizer = tf.train.sgd(learningRate); //construindo um otimizador que utiliza a descida do gradiente

function setup() { //declarando uma função
    createCanvas(400,400); //definindo o tamanho do sketch
    m = tf.variable(tf.scalar(random(1))); //criando uma variavel global m que armazena/cria uma escalar aleatória
    b = tf.variable(tf.scalar(random(1))); //criando uma variavel global b que armazena/cria uma escalar aleatória
}

function loss(pred, labels) { //declarando a função de erro
    return pred //retornando a previsao
    .sub(labels) //subtraindo os valores reais das adivinhações
    .square() //elevando ao quadrado os valores anteriores
    .mean(); //fazendo a média dos valores
}

function predict(x) { //declarando a função de previsão do x
    const xs = tf.tensor1d(x); //constante xs que cria um tensor de 1 dimensão a partir de x, é o novo valor de x
    const ys = xs.mul(m).add(b); //literalmente, y = mx + b
    return ys; //retornando a previsao, que é ys
}

function mousePressed() { //execução em função do mouse
    let x = map(mouseX, 0, width, 0, 1); 
    let y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y); //isso tudo aqui, é pra adicionar pontos conforme o mouse clicar no gráfico, em um array
}

function draw() { //função para desenhar
    tf.tidy(() => { //define que os valores não são permanentes
        if (x_vals.length > 0) { //se os valores de x vezes o comprimento forem maiores do que 0
            const ys = tf.tensor1d(y_vals); //ys é o novo valor de y, adicionado em um tensor de uma dimensão
            optimizer.minimize(() => loss(predict(x_vals), ys)); //minimizando o erro
        }
    });

    background(0); //fundo

    stroke(255); 
    strokeWeight(8); //desenha o tamanho das bolinhas do gráfico, ambos os 'stroke'
    for (let i = 0; i < x_vals.length; i++) {
        let px = map(x_vals[i], 0, 1, 0, width);
        let py = map(y_vals[i], 0, 1, height, 0);
        point(px,py); //essa parte da função vai adicionar os pontos clicados no gráfico
    }

    const lineX = [0,1]; //constante de linha x com uma lista de valores 0 e 1

    const ys = tf.tidy(() => predict(lineX)); //chamando a previsão da linha x
    let lineY = ys.dataSync(); //adiciona os valores de ys à linha y
    ys.dispose(); //liberar a memória de ys

    let x1 = map(lineX[0], 0, 1, 0, width);
    let x2 = map(lineX[1], 0, 1, 0, width);

    let y1 = map(lineY[0], 0, 1, height, 0);
    let y2 = map(lineY[1], 0, 1, height, 0);

    strokeWeight(2);
    line(x1,y1,x2,y2); //as linhas 56 à 66 desenham a linha do gráfico da regressão

    console.log(tf.memory().numTensors); //para executar o gráfico dos específicos pontos recebidos no momento
}