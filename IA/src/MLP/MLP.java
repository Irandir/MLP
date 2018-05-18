package MLP;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import grafico.Grafico;

public class MLP {
	static Random rand = new Random();

	public static int porcentagemDoDados(int tamanhoDosDados, double por) {
		int i = 0;
		i = (int) (tamanhoDosDados * por);
		return i;
	}

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

	public static double normazindoDado(double dadoNormal, double min, double max) {
		return (dadoNormal - min) / (max - min);
	}

	private static double desnormalizar(double dadoNormal, double min, double max) {
		return dadoNormal * (max - min) + min;
	}

	
	
	public static void main(String[] args) {
		
		int tamanhoDaJanela = 12;
		// n --> taxa de aprendizagem
		double n = 0.2;

		// qNE é o numero de neuronios da Camada de Escondida
		int qNE = 10;

		// qNS é o numero de neuronios da Camada de Saida
		int qNS = 1;

		// numero de epocas do treinamento
		int epocas = 1000;
		String nomeDaBase = "leite.txt";
		List<Double> dados = leituraDeArquivo(nomeDaBase);
		double min = dados.get(0);
		double max = dados.get(0);
		for (int i = 0; i < dados.size(); i++) {
			if (min > dados.get(i)) {
				min = dados.get(i);
			}
			if (max < dados.get(i)) {
				max = dados.get(i);
			}
		}
		int dadosTamanho = dados.size()-tamanhoDaJanela;
		
		// tamaho Do Conjunto de Treino
		int treinoLegth = porcentagemDoDados(dadosTamanho, 0.5);

		// tamaho Do Conjunto de validacao
		int validaLegth = porcentagemDoDados(dadosTamanho, 0.2);

		// tamaho Do Conjunto de validacao
		int testeLegth = porcentagemDoDados(dadosTamanho, 0.3);

		double entrada[][] = new double[treinoLegth][tamanhoDaJanela];

		double saida[] = new double[treinoLegth];

		// colocando os dados
		for (int i = 0; i < entrada.length; i++) {
			for (int j = 0; j < entrada[0].length; j++) {
				entrada[i][j] = dados.get(j + i);
			}
			saida[i] = dados.get(entrada[0].length + i);

		}

		// entrada Normalizada
		double entradaN[][] = new double[entrada.length][entrada[0].length];

		// saida Normalizada
		double saidaN[] = new double[saida.length];
		double treino[]  = new double[saida.length];
		// colocando os dados Normalizados
		for (int i = 0; i < entradaN.length; i++) {
			for (int j = 0; j < entradaN[0].length; j++) {
				entradaN[i][j] = normazindoDado(entrada[i][j], min, max);
			}
		}
		for (int i = 0; i < saidaN.length; i++) {
			saidaN[i] = normazindoDado(saida[i], min, max);
		}

		// pesos da camada entrada
		double pesos[][] = gerandoPesosAleatorio(qNE, entradaN[0].length);

		// bias da camada de entrada
		double bias[] = gerandoPesosAleatorio(qNE);

		// NET(s) da camada escondida
		double nets[] = new double[qNE];
		double fnets[] = new double[qNE];

		// pesos da camada escondida
		double pesosE[][] = gerandoPesosAleatorio(qNS, fnets.length);

		// bias da camada de escondida
		double biasE[] = gerandoPesosAleatorio(qNS);

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

		double erroEpocas[] = new double[entradaN.length];

		// var auxliar para soma
		double s = 0;

		double saidaDesnormalizada = 0;// saida desnormalizada
		double p = 0;
		// for das epocas do treinamento
		for (int contEpocas = 0; contEpocas < epocas; contEpocas++) {

			// cada instancia (entrada)
			for (int y = 0; y < entradaN.length; y++) {

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
						nets[i] += pesos[i][j] * entradaN[y][j];
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
					fnetsO[i] = netsO[i];

				}
				treino[y] = desnormalizar(fnetsO[0], min, max);
				// System.out.println("fnetO = " + fnetsO[0] + " ideal -->" +
				// saidaN[y]);

				// calculando os erros
				for (int i = 0; i < erro.length; i++) {
					erro[i] = saidaN[y] - fnetsO[0];
				}

				// calculando o grandiete 0
				for (int i = 0; i < g0.length; i++) {
					g0[i] = erro[i];
					//g0[i] = erro[i] * fnetsO[i] * (1 - fnetsO[i]);
					// System.out.println("g0 -->"+ g0[i]);
				}

				// calculando o grandiete H
				for (int i = 0; i < gH.length; i++) {

					s = 0;

					for (int j = 0; j < g0.length; j++) {
						s += g0[j] * pesosE[j][i];
					}

					gH[i] = fnets[i] * (1 - fnets[i]) * s;

				}

				// atualização dos pesos (Demonio)
				for (int i = 0; i < pesosE.length; i++) {
					for (int j = 0; j < pesosE[0].length; j++) {
						pesosE[i][j] = pesosE[i][j] + n * g0[i] * fnets[j];
					} // bias E
					biasE[i] = biasE[i] + n * g0[i] * 1;

				}

				for (int i = 0; i < pesos.length; i++) {
					for (int j = 0; j < pesos[0].length; j++) {
						pesos[i][j] = pesos[i][j] + n * gH[i] * entradaN[y][j];
					}
				}

				// atuaizar o bias
				for (int i = 0; i < bias.length; i++) {
					bias[i] = bias[i] + n * gH[i] * 1;
				}

				saidaDesnormalizada = desnormalizar(fnetsO[0], min, max);

				// EQM
				erroEpocas[y] = Math.pow((saida[y] - saidaDesnormalizada), 2);

			} // final da interacao da instancia

			// EQM
			custo = 0;
			for (int i = 0; i < erroEpocas.length; i++) {
				custo += erroEpocas[i];
			}
			custo /= entradaN.length;

			System.out.println("epoca --> " + contEpocas + " custo --> " + custo
					+ "\n____________________________________________________");
			if (contEpocas == 0) {
				p = custo;
			}
		} // final das epocas
		System.out.println("epoca 0  custo --->" + p);

		// =========================================validação=====================================
		double entradaV[][] = new double[validaLegth][tamanhoDaJanela];
		double saidaV[] = new double[validaLegth];

		// colocando os dados
		for (int i = 0; i < entradaV.length; i++) {
			for (int j = 0; j < entradaV[0].length; j++) {
				entradaV[i][j] = dados.get(j + i + entrada.length);
			}
			saidaV[i] = dados.get(entradaV[0].length + i + entrada.length);
		}

		// entrada Normalizada
		double entradaVN[][] = new double[entradaV.length][entradaV[0].length];

		// saida Normalizada
		double saidaVN[] = new double[saidaV.length];
		double validacao[]  = new double[saidaV.length];
		// colocando os dados Normalizados
		for (int i = 0; i < entradaVN.length; i++) {
			for (int j = 0; j < entradaVN[0].length; j++) {
				entradaVN[i][j] = normazindoDado(entradaV[i][j], min, max);
			}
		}
		for (int i = 0; i < saidaVN.length; i++) {
			saidaVN[i] = normazindoDado(saidaV[i], min, max);
		}

		double eqm = 0;
		double saidaVDesejada = 0;
		double saidaVObtida = 0;

		System.out.println("________________________Validacao____________________");
		for (int c = 0; c < entradaVN.length; c++) {

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

			for (int i = 0; i < nets.length; i++) {

				for (int j = 0; j < pesos[0].length; j++) {
					nets[i] += pesos[i][j] * entradaVN[c][j];
				}
				nets[i] += bias[i];
				fnets[i] = f(nets[i]);

			}
			for (int i = 0; i < netsO.length; i++) {

				for (int j = 0; j < nets.length; j++) {
					netsO[i] += fnets[j] * pesosE[i][j];
				}
				netsO[i] += biasE[i];
				fnetsO[i] = netsO[i];
			}
			saidaVDesejada = desnormalizar(fnetsO[0], min, max);
			validacao[c] = saidaVDesejada;
			saidaVObtida = desnormalizar(saidaVN[c], min, max);
			// System.out.println("-->" + fnetsO[0] + " "+saidaVN[c]+" "+
			// desnormalizar(fnetsO[0], min, max)+ " "+desnormalizar(saidaVN[c],
			// min, max));
			// EQM
			eqm += Math.pow((saidaVDesejada - saidaVObtida), 2);
		}
		eqm /= saidaV.length;
		System.out.println("custo de valição --> " + eqm);

		// =======================================================teste============================

		// entrada teste
		double entradaT[][] = new double[testeLegth][tamanhoDaJanela];
		double saidaT[] = new double[testeLegth];

		// colocando os dados
		for (int i = 0; i < entradaT.length; i++) {
			
			for (int j = 0; j < entradaT[0].length; j++) {
				entradaT[i][j] = dados.get(j + i + entrada.length+entradaV.length);
			}
			saidaT[i] = dados.get(entradaT[0].length + i + entrada.length+entradaV.length);
		}

		// entrada Normalizada
		double entradaTN[][] = new double[entradaT.length][entradaT[0].length];

		// saida Normalizada
		double saidaTN[] = new double[saidaT.length];
		double teste[] = new double[saidaT.length];
		// colocando os dados Normalizados
		for (int i = 0; i < entradaTN.length; i++) {
			for (int j = 0; j < entradaTN[0].length; j++) {
				entradaTN[i][j] = normazindoDado(entradaT[i][j], min, max);
			}
		}
		for (int i = 0; i < saidaTN.length; i++) {
			saidaTN[i] = normazindoDado(saidaT[i], min, max);
		}

		eqm = 0;
		saidaVDesejada = 0;
		saidaVObtida = 0;

		System.out.println("________________________teste____________________");
		for (int c = 0; c < entradaTN.length; c++) {

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

			for (int i = 0; i < nets.length; i++) {

				for (int j = 0; j < pesos[0].length; j++) {
					nets[i] += pesos[i][j] * entradaTN[c][j];
				}
				nets[i] += bias[i];
				fnets[i] = f(nets[i]);

			}
			for (int i = 0; i < netsO.length; i++) {

				for (int j = 0; j < nets.length; j++) {
					netsO[i] += fnets[j] * pesosE[i][j];
				}
				netsO[i] += biasE[i];
				fnetsO[i] = netsO[i];
			}
			
			saidaVDesejada = desnormalizar(fnetsO[0], min, max);
			teste[c] = saidaVDesejada;
			saidaVObtida = desnormalizar(saidaTN[c], min, max);
			//System.out.println("-->" + fnetsO[0] + " " + saidaTN[c] + " " + desnormalizar(fnetsO[0], min, max) + " "
				//	+ desnormalizar(saidaTN[c], min, max));
			
			// EQM
			eqm += Math.pow((saidaVDesejada - saidaVObtida), 2);
		}
		eqm /= saidaT.length;
		System.out.println("custo de teste --> " + eqm);

		Grafico g = new Grafico();
		g.mostrar2(saida, saidaV, saidaT, treino, validacao, teste, "MLP", nomeDaBase);
		
	}// final da class

	private static double f(double d) {

		return 1.0 / (1 + Math.exp(-d));

	}

	public static ArrayList<Double> leituraDeArquivo(String nome) {

		BufferedReader br = null;
		ArrayList<Double> dados = new ArrayList<Double>();
		try {

			String path = MLP.class.getResource(nome).getPath();
			br = new BufferedReader(new FileReader(path));
			int aux = 0;
			String linha = null;
			while ((linha = br.readLine()) != null) {
				dados.add(Double.parseDouble(linha));

			}

			br.close();
			return dados;
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return null;
	}
}
