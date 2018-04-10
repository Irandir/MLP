package MPL;

import java.util.Random;

public class MPL {
	static Random rand = new Random();

	// gera vaolores aleatorios entre 0 e 1
	public static double[][] gerandoPesosAleatorio(int qN, int n) {
		double pesos[][] = new double[qN][n];

		for (int i = 0; i < pesos.length; i++) {
			for (int j = 0; j < pesos[0].length; j++) {
				pesos[i][j] = rand.nextDouble();
			}
		}

		return pesos;
	}

	// gera vaolores aleatorios entre 0 e 1
	public static double[] gerandoPesosAleatorio(int qN) {
		double pesos[] = new double[qN];

		for (int i = 0; i < pesos.length; i++) {
			pesos[i] = rand.nextDouble();
		}

		return pesos;
	}

	public static void main(String[] args) {
		// entrada
		int entrada[][] = { { 0, 0}, { 0, 1}, { 1, 0}, { 1, 1} };

		// saida XOR
		int saida[] = { 0, 1, 1, 0 };

		// n --> taxa de aprendizagem
		double n = 0.1;
		
		// qNE é o numero de neuronios da Camada de Escondida
		int qNE = 3;

		// qNS é o numero de neuronios da Camada de Saida
		int qNS = 1;

		// numero de epocas do treinamento
		int epocas = 100;

		// pesos da camada entrada
		double pesos[][] = {{0.1,0.2},{0.1,0.2},{0.1,0.2}};

		// bias da camada de entrada
		double bias[] = {0.1,0.2,0.3};

		// NET(s) da camada escondida
		double nets[] = new double[qNE];
		double fnets[] = new double[qNE];
		
		// pesos da camada escondida
		double pesosE[][] = {{0.2,0.1,0.2}};

		// bias da camada de escondida
		double biasE[] = {0.1};

		// saidas obitidas
		double netsO[] = new double[qNS];
		double fnetsO[] = new double[qNS];

		// variavel para guarda a função de custo
		double custo = 0;

		// Gradiente 0(Gradiente Saida)
		double[] g0 = new double[qNS];

		// Gradiente H(Gradiente Layer escondida)
		double[] gH = new double[nets.length];

		// erro
		double erro[] = new double[qNS];

		//var auxliar para soma 
		double s = 0;
		
		// for das epocas do treinamento
		for (int contEpocas = 0; contEpocas < epocas; contEpocas++) {

			// cada instancia (entrada)
			for (int y = 0; y < entrada.length; y++) {

				// resetando a saida
				for (int i = 0; i < netsO.length; i++) {
					netsO[i] = 0;
					fnetsO[i] = 0;
				}
				// resert NET(s)
				for (int i = 0; i < nets.length; i++) {
					nets[i] = 0;
					fnets[i] = 0;
				}

				// resert g0
				for (int i = 0; i < g0.length; i++) {
					g0[i] = 0;
				}

				// resert erro
				for (int i = 0; i < erro.length; i++) {
					erro[i] = 0;
				}

				// resert gH 
				for (int i = 0; i < gH.length; i++) {
					gH[i] = 0;
				}
				
				// alimentando os NETs (NET1 = Somatorio dos pesos * entradas)
				for (int i = 0; i < nets.length; i++) {

					for (int j = 0; j < pesos[0].length; j++) {
						nets[i] += pesos[i][j] * entrada[y][j];
					}
					nets[i] += bias[i];
					fnets[i] = f(nets[i]);
					

				}
				
				// alimentando a(s) saida(s)
				for (int i = 0; i < netsO.length; i++) {

					for (int j = 0; j < nets.length; j++) {
						netsO[i] += fnets[j] * pesosE[i][j];
					}
					netsO[i] += biasE[i];
					fnetsO[i] = f(netsO[i]);
					
				}
				
				
				System.out.println("fnetO = "+fnetsO[0] +" ideal -->"+saida[y]);
				//-----------------------------------------------------------------------------------------------------------------
				
				// calculando os erros
				for (int i = 0; i < erro.length; i++) {
					erro[i] = saida[i] - fnetsO[i];
				}

				// calculando o grandiete 0
				for (int i = 0; i < g0.length; i++) {
					g0[i] = erro[i] * fnetsO[i] * (1 - fnetsO[i]);
					//System.out.println("g0 -->"+ g0[i]);
				}
				
				// calculando o grandiete H 
				for (int i = 0; i < gH.length; i++) {
					
					s = 0;
					
					for (int j = 0; j < g0.length; j++) {
						s+= g0[j] * pesosE[j][i];
					}
					
					gH[i] = fnets[i] * (1 - fnets[i]) * s;
					
				}

				//atualização dos pesos (Demonio)
				for (int i = 0; i < pesosE.length; i++) {
					for (int j = 0; j < pesosE[0].length; j++) {
						pesosE[i][j] = pesosE[i][j] + n * g0[i] * fnets[j];
					}//bias E
					biasE[i] = biasE[i] + n *g0[i] * 1;

				}
				
				for (int i = 0; i < pesos.length; i++) {
					for (int j = 0; j < pesos[0].length; j++) {
						pesos[i][j] = pesos[i][j] + n * gH[i] * entrada[y][j];
					}
				}
				
				//atuaizar o bias
				for (int i = 0; i < bias.length; i++) {
					bias[i] = bias[i] + n *gH[i] * 1;
				}
				custo += Math.pow(erro[0],2); 
	
			}
			System.out.println("custo --> "+custo/4+"\n____________________________________________________");
		}
		System.out.println();
	}

	private static double f(double d) {

		return 1.0 / (1 + Math.exp(-d));
		
	}
}
