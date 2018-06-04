function main
	[P,T]=obtenerDataset;%dataset
	[vcn,vtf]=obtenerArquitectura;%arquitectura
	[W1,B1,W2,B2]=initWaB;%inicialización de pesos y bias
	[alpha,maxepoch,minEtrain,valepoch,numval]=obtenerDatos;%datos de validación
	mlp(P,T,vcn,vtf,alpha,maxepoch,minEtrain,valepoch,numval,w1,b1,w2,b2)%llama a la red
end
function [P,T]=obtenerDataset
	P=-2:.2:2;
	T=1+sin((pi/4)*P);
end
function [vcn,vtf]=obtenerArquitectura
	vcn=[1 2 1];
	vtf=[2 1];
end
function [W1,B1,W2,B2]=initWaB
	W1=[-.27;-.41];
	B1=[-.48;-.13];
	W2=[.09 -.17];
	B2=.48;
end
function [alpha,maxepoch,minEtrain,valepoch,numval]=obtenerDatos;
	alpha=.1;
	maxepoch=5;
	minEtrain=.0001;
	valepoch=10;
	numval=3;
end
function mlp(P,T,vcn,vtf,alpha,maxepoch,minEtrain,valepoch,numval,w1,b1,w2,b2)
	a=feedforward(w,b,vtf,p);
	[w1,b1]=backpropagation(a,w1,b1,functions,e,alpha)
end
function [a] = feedforward(w,b,functions,p)
	a = {}
	a{1} = p
	for cont = 1:length(w)
		switch functions(cont)
			case 1
				a{cont+1}=purelin(w{cont}*a{cont}+b{cont})
			case 2
				a{cont+1}=logsig(w{cont}*a{cont}+b{cont})
			case 3
				a{cont+1}=tansig(w{cont}*a{cont}+b{cont})
		end
	end
end
function [w, b] = backpropagation(a,w,b,functions,e,alpha)
	sensitivities = {}
	sensitivities{length(functions)} = -2*e
	cont = length(functions) - 1
	while cont >= 1
		if functions{cont} == 2
			sensitivities{cont} = diag(a{cont+1}*diag(1-a{cont+1}))*w{cont+1}'*sensitivities{cont+1}
		else
			sensitivities{cont} = diag(1-a{cont+1}^2)*w{cont+1}'*sensitivities{cont+1}
		end
		cont = cont-1
	end

	for cont = 1:length(w)
		w{cont} = w{cont}-alpha*sensitivities{cont}*a{cont}'
		b{cont} = b{cont}-alpha*sensitivities{cont}
	end 
end
