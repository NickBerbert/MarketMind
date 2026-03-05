import plotly.graph_objects as go


def criar_grafico_com_previsao(historico, ticker, previsoes_ensemble=None, datas_previsao=None, detalhes_previsoes=None):
    fig = go.Figure()

    if historico is not None and not historico.empty:
        # Linha histórica
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            mode='lines',
            name='Histórico Real',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>R$ %{y:.2f}</b><br>%{x}<extra></extra>'
        ))

        # Previsões
        if previsoes_ensemble is not None and datas_previsao is not None:
            # Bandas de confiança (se disponíveis)
            if detalhes_previsoes and 'lower_bound' in detalhes_previsoes and 'upper_bound' in detalhes_previsoes:
                lower_bound = detalhes_previsoes['lower_bound']
                upper_bound = detalhes_previsoes['upper_bound']

                # Banda superior
                fig.add_trace(go.Scatter(
                    x=datas_previsao,
                    y=upper_bound,
                    mode='lines',
                    name='IC 95% Superior',
                    line=dict(color='rgba(255,68,68,0.3)', width=1),
                    showlegend=True,
                    hovertemplate='<b>Limite Superior</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
                ))

                # Banda inferior
                fig.add_trace(go.Scatter(
                    x=datas_previsao,
                    y=lower_bound,
                    mode='lines',
                    name='IC 95% Inferior',
                    line=dict(color='rgba(255,68,68,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,68,68,0.2)',
                    showlegend=True,
                    hovertemplate='<b>Limite Inferior</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
                ))

            # Previsão média (linha principal)
            fig.add_trace(go.Scatter(
                x=datas_previsao,
                y=previsoes_ensemble,
                mode='lines+markers',
                name='Previsão GRU (Média)',
                line=dict(color='#ff4444', width=3),
                marker=dict(color='#ff4444', size=8),
                hovertemplate='<b>GRU</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
            ))

            # Linha conectora
            ultimo_preco = historico['Close'].iloc[-1]
            ultima_data = historico.index[-1]

            fig.add_trace(go.Scatter(
                x=[ultima_data, datas_previsao[0]],
                y=[ultimo_preco, previsoes_ensemble[0]],
                mode='lines',
                line=dict(color='#ff4444', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='<extra></extra>'
            ))

    # Layout do gráfico
    fig.update_layout(
        title=f'{ticker} - Análise e Previsão',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='#00d4ff',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
    )

    return fig
