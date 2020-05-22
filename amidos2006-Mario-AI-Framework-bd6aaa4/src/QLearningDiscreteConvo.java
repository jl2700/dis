import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class QLearningDiscreteConvo<O extends Encodable> extends QLearningDiscrete<O> {

	public QLearningDiscreteConvo(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
			int epsilonNbStep) {
		super(mdp, dqn, conf, epsilonNbStep);
	}

	public QLearningDiscreteConvo(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn,
			QLConfiguration conf, IDataManager dataManager) {
		this(mdp, dqn, conf, conf.getEpsilonNbStep());
		addListener(new DataManagerTrainingListener(dataManager));
	}

	public QLearningDiscreteConvo(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdConv.Configuration netConf,
			 QLConfiguration conf, IDataManager dataManager) {
		this(mdp, buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize(), netConf), conf, dataManager);
	}
	
	public static DQN buildDQN(int shapeInputs[], int numOutputs, DQNFactoryStdConv.Configuration conf) {

        if (shapeInputs.length == 1)
            throw new AssertionError("Impossible to apply convolutional layer on a shape == 1");


        NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(Constants.NEURAL_NET_SEED)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .l2(conf.getL2())
                        .updater(conf.getUpdater() != null ? conf.getUpdater() : new Adam())
                        .weightInit(WeightInit.XAVIER).l2(conf.getL2()).list()
                        .layer(0, new ConvolutionLayer.Builder(8, 8).nIn(shapeInputs[0]).nOut(16).stride(4, 4)
                                        .activation(Activation.RELU).build());


        confB.layer(1, new ConvolutionLayer.Builder(4, 4).padding(2, 2).nOut(32).stride(2, 2).activation(Activation.RELU).build());

        confB.layer(2, new DenseLayer.Builder().nOut(256).activation(Activation.RELU).build());

        confB.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nOut(numOutputs)
                        .build());

        confB.setInputType(InputType.convolutional(shapeInputs[1], shapeInputs[2], shapeInputs[0]));
        MultiLayerConfiguration mlnconf = confB.build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        if (conf.getListeners() != null) {
            model.setListeners(conf.getListeners());
        } else {
            model.setListeners(new ScoreIterationListener(Constants.NEURAL_NET_ITERATION_LISTENER));
        }

        return new DQN(model);
    }

}
