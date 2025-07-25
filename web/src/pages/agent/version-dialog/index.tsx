import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Spin } from '@/components/ui/spin';
import {
  useFetchVersion,
  useFetchVersionList,
} from '@/hooks/use-agent-request';
import { IModalProps } from '@/interfaces/common';
import { cn } from '@/lib/utils';
import { formatDate } from '@/utils/date';
import { downloadJsonFile } from '@/utils/file-util';
import {
  Background,
  ConnectionMode,
  ReactFlow,
  ReactFlowProvider,
} from '@xyflow/react';
import { ArrowDownToLine } from 'lucide-react';
import { ReactNode, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { nodeTypes } from '../canvas';

export function VersionDialog({
  hideModal,
}: IModalProps<any> & { initialName?: string; title?: ReactNode }) {
  const { t } = useTranslation();
  const { data, loading } = useFetchVersionList();
  const [selectedId, setSelectedId] = useState<string>('');
  const { data: agent, loading: versionLoading } = useFetchVersion(selectedId);

  const handleClick = useCallback(
    (id: string) => () => {
      setSelectedId(id);
    },
    [],
  );

  const downloadFile = useCallback(() => {
    const graph = agent?.dsl.graph;
    if (graph) {
      downloadJsonFile(graph, agent?.title);
    }
  }, [agent?.dsl.graph, agent?.title]);

  useEffect(() => {
    if (data.length > 0) {
      setSelectedId(data[0].id);
    }
  }, [data]);

  return (
    <Dialog open onOpenChange={hideModal}>
      <DialogContent className="max-w-[60vw]">
        <DialogHeader>
          <DialogTitle>{t('flow.historyversion')}</DialogTitle>
        </DialogHeader>
        <section className="flex gap-2 relative">
          <div className="w-1/3 max-h-[60vh] overflow-auto min-h-[40vh]">
            {loading ? (
              <Spin className="top-1/2"></Spin>
            ) : (
              <ul className="space-y-2">
                {data.map((x) => (
                  <li
                    key={x.id}
                    className={cn('cursor-pointer', {
                      'bg-card rounded p-1': x.id === selectedId,
                    })}
                    onClick={handleClick(x.id)}
                  >
                    {x.title}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="relative flex-1 ">
            {versionLoading ? (
              <Spin className="top-1/2" />
            ) : (
              <Card className="h-full">
                <CardContent className="h-full p-5">
                  <section className="flex justify-between">
                    <div>
                      <div className="pb-1">{agent?.title}</div>
                      <p className="text-text-sub-title text-xs">
                        {formatDate(agent?.create_date)}
                      </p>
                    </div>
                    <Button variant={'ghost'} onClick={downloadFile}>
                      <ArrowDownToLine />
                    </Button>
                  </section>
                  <ReactFlowProvider key={`flow-${selectedId}`}>
                    <ReactFlow
                      connectionMode={ConnectionMode.Loose}
                      nodes={agent?.dsl.graph?.nodes || []}
                      edges={
                        agent?.dsl.graph?.edges.flatMap((x) => ({
                          ...x,
                          type: 'default',
                        })) || []
                      }
                      fitView
                      nodeTypes={nodeTypes}
                      edgeTypes={{}}
                      zoomOnScroll={true}
                      panOnDrag={true}
                      zoomOnDoubleClick={false}
                      preventScrolling={true}
                      minZoom={0.1}
                    >
                      <Background />
                    </ReactFlow>
                  </ReactFlowProvider>
                </CardContent>
              </Card>
            )}
          </div>
        </section>
      </DialogContent>
    </Dialog>
  );
}
