import * as tf from '@tensorflow/tfjs';

require('@tensorflow/tfjs-node');


console.log('Project initialisation....');
console.log('Project initialisation  two.');


//linear regression
const linearModel = tf.sequential();
//add model configs
linearModel.add(tf.layers.dense({
    units: 1, inputShape: [1]
}));

//set model optimizer
linearModel.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
});

//training set ****************************************************************
const xLineSet = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const yLineSet = tf.tensor2d([1, 3, 5, 7], [4, 1]);


//train data ******************************************************************
const  trainModel = async (xs, ys, model) => {
    console.log('model training ....');
    return await model.fit(xs, ys, {epochs: 20});
};


//predict data ****************************************************************
const predict = (model) => {
    console.log('model prediction ....');
    model.predict(tf.tensor2d([5], [1, 1])).print();
};


// RUN APP ********************************************************************

const appRun = async (xSet, ySet) => {
    console.log('app run ....');
    await trainModel(xSet, ySet, linearModel);
    predict(linearModel);

};

appRun(xLineSet, yLineSet);

// save model *****************************************************************
